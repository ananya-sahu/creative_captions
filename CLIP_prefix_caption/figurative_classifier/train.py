"""Training and evaluation functions for the metaphor & novelty project."""

import gc
import copy
import logging
import numpy as np
import torch

from torch.nn import MSELoss, BCELoss, NLLLoss
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW
from evaluate import evaluate



def train(model, train, dev, train_steps, lr, metaphor_weight, alpha, beta,
          **kwargs):
    """
    Train MetaphorModel on Metaphor data.
    Args:
        model (nn.Module): initialised model, untrained
        train (DataLoader): object containing metaphor training data.
        dev (DataLoader): object containing metaphor validation data.
        train_steps (int): number of updates to train for
        lr (float): learning rate for optimiser
        metaphor_weight (float): weight for the positive metaphor class
        alpha (float): weight for the metaphor detection task
        beta (float): weight for the novelty prediction task
    Returns:
        best_model: state_dict of the best model according to validation data.
    """
    # Optimiser
    optimiser = AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=int(train_steps * .1), num_training_steps=train_steps
    )
    train_iter = iter(train)
    loss_fn_novelty = MSELoss()
    # loss_fn_metaphor = NLLLoss(ignore_index=-2, weight=torch.FloatTensor([1 - metaphor_weight, metaphor_weight]))
    metaphor_losses, novelty_losses = [], []

    for x in range(train_steps):
        try:
            batch = train_iter.next()
        except StopIteration:
            train_iter = iter(train)
            metaphor_losses = []
            novelty_losses = []
            batch = train_iter.next()

        # Forward pass through the model
        model.train()
        optimiser.zero_grad()
        metaphoricity_output, novelty_output = model(batch.tokens, batch.mask)

        # Compute loss for metaphoricity labels
        metaphoricity_labels = batch.bert_metaphoricity_labels.view(-1)
        metaphoricity_output = metaphoricity_output.cpu().contiguous().view(-1)
        weights = copy.deepcopy(metaphoricity_labels)
        weights[weights == 1] = metaphor_weight
        weights[weights == 0] = 1 - metaphor_weight                
        weights[weights == -2] = 0
        loss_fn_metaphor = BCELoss(weight=weights)
        metaphoricity_loss = loss_fn_metaphor(
            metaphoricity_output,
            metaphoricity_labels)
        metaphor_losses.append(metaphoricity_loss.item())

        # # Compute loss for novelty scores
        novelty_labels = batch.bert_novelty_labels.view(-1)
        novelty_output = novelty_output.cpu().contiguous().view(-1)
        weights = copy.deepcopy(novelty_labels)
        weights[weights != -2] = 1
        weights[weights == -2] = 0                
        novelty_loss = loss_fn_novelty(novelty_output * weights, novelty_labels * weights)
        novelty_losses.append(novelty_loss.item())

        loss = alpha * metaphoricity_loss + beta * novelty_loss
        loss.backward()
        optimiser.step()
        scheduler.step()
        torch.cuda.empty_cache()

        # Evaluation loop
        if (x + 1) % 500 == 0:
            logging.info(f"Training, Metaphor Loss: {np.mean(metaphor_losses):.3f} " + \
                         f"Novelty Loss: {np.mean(novelty_losses):.3f}")
            metaphoricity_score, novelty_score = evaluate(model, dev)
            #score_devs.append(score_dev)
            torch.cuda.empty_cache()
    return model


def clean_object_from_memory(obj):
    """
    Clean Pytorch object from memory.
    Args:
        obj: Pytorch object
    """
    del obj
    gc.collect()
    torch.cuda.empty_cache()