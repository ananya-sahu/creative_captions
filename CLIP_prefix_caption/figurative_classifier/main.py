"""Main functionality for the metaphor & novelty project."""

import os
import random
import logging
import argparse
import torch
import numpy as np

from data import load_data
from train import train
from evaluate import evaluate
from model import MetaphorModel

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    logging.info(f"Seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()

    # Task independent arguments related to preprocessing or training
    group = parser.add_argument_group("training")
    group.add_argument("--lr", type=float, default=3e-5)
    group.add_argument("--seed", type=int, default=1)
    group.add_argument("--metaphor_weight", type=float, default=0.8)
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--train_steps", type=int, default=1000)
    group.add_argument("--output", type=str, default="output.tsv")
    group.add_argument("--alpha", type=float, default=0.5)
    group.add_argument("--beta", type=float, default=0.5)
    args = vars(parser.parse_args())

    for key, value in args.items():
        logging.info(f"--{key}: {value}")

    # Set seed to combat random effects
    set_seed(args["seed"])

    train_set, dev, test = load_data(args["batch_size"])

    # Initialise an empty model and train it.
    model = MetaphorModel()
    if torch.cuda.is_available():
        model.cuda()
    # Evaluate every epoch on the validation data.
    best_model = train(model, train_set, dev, **args)

    # Evaluate the trained model on test data.
    logging.info("Best Model")
    evaluate(best_model, test, "test", args["output"])
    torch.save(best_model, "model.pt")