"""Inference functionality for the metaphor & novelty project."""

import os
import logging
import argparse
import csv
import torch


from torch.utils.data import DataLoader, SequentialSampler
from data import CustomDataset
from evaluate import evaluate
from models import MetaphorModel

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()

    # Task independent arguments related to preprocessing or training
    group = parser.add_argument_group("model")
    group.add_argument("--model_name", type=str, default="model.pt")
    group.add_argument("--output", type=str, default="output_hippo.tsv")
    args = vars(parser.parse_args())
    logging.info(args)

    # Set seed to combat random effects
    model = torch.load(args["model_name"])

    snts, metaphoricity_labels, novelty_labels = [], [], []
    with open("data/met_hippocorpus.csv") as csvfile:
        corpus = csv.reader(csvfile, delimiter=',')
        corpus.readline()
        for row in corpus:
            sentence = row[0].split()
            snts.append(sentence)
            metaphoricity_labels.append([0 for _ in sentence])
            novelty_labels.append([0 for _ in sentence])

    dataset = CustomDataset(
        list(snts), list(metaphoricity_labels),
        list(novelty_labels))

    sampler = SequentialSampler(dataset)
    hippocorpus = DataLoader(
        dataset, batch_size=32, sampler=sampler,
        collate_fn=CustomDataset.collate_fn
    )

    # Evaluate the trained model on test data.
    evaluate(model, hippocorpus, output_filename=args["output"], no_labels=True)