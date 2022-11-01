"""
Data functions for the metaphor & novelty project.
Create a TextDataset object to gather data samples and turn them into batches.
"""

import csv
import codecs
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, \
    RandomSampler, SequentialSampler
from transformers import BertTokenizer

TOKENIZER = BertTokenizer.from_pretrained("bert-base-cased")


class Batch:
    """Batch object."""

    def __init__(self, sentences, lengths, mask, tokens, mapping,
                 bert_metaphoricity_labels, bert_novelty_labels,
                 metaphoricity_labels, novelty_labels):
        """Initialise Batch object.
        Args:
            src (list of str): source sentence
            lengths (list): lengths of tokenised sentences
            mask (list): mask corresponding to src_tokens
            tokens (list): BERTTOKENIZER token ids
            mapping (dict): maps BERT token ids back to original word ids
            metaphoricity_labels (list): scores -2, 0, 1 for metaphoricity
            novelty_labels (list): scores -2 or range -1 to 1 for novelty
        """
        # Standard focus sentence tensors
        self.sentences = sentences
        self.lengths = torch.LongTensor(lengths)
        self.mask = torch.FloatTensor(mask)

        # BERT specific tensor
        self.mapping = mapping
        self.tokens = torch.LongTensor(tokens)
        self.bert_metaphoricity_labels = torch.FloatTensor(bert_metaphoricity_labels)
        self.bert_novelty_labels = torch.FloatTensor(bert_novelty_labels)
        self.metaphoricity_labels = metaphoricity_labels
        self.novelty_labels = novelty_labels


class CustomDataset(Dataset):
    """Dataset object."""

    def __init__(self, texts, metaphoricity_labels, novelty_labels):
        self.texts = texts
        self.metaphoricity_labels = metaphoricity_labels
        self.novelty_labels = novelty_labels

    def __getitem__(self, idx):
        """Extract one sample with index idx.
        Args:
            idx (int): sample number
        Returns:
            text (list of str): words
            label (int): 0 or 1, label
        """
        text = self.texts[idx]
        metaphoricity_label = self.metaphoricity_labels[idx]
        novelty_label = self.novelty_labels[idx]
        return text, metaphoricity_label, novelty_label

    def __len__(self):
        """Compute number of samples in dataset."""
        return len(self.texts)

    @staticmethod
    def collate_fn(batch):
        """Gather the batches' text, lengths, labels and masks.
        Args:
            batch (list of tuples): texts, lengths and labels
        Returns:
            batch_text (list of list of str): words
            lengths (LongTensor): vector with sentence lengths
            labels (LongTensor): binary labels of metaphoricity
            mask (LongTensor): binary indications of words vs padding
        """
        def bert_process_sentences(sentences, labels):
            # Retokenise using BERT-specific tokenizer
            bert_sentences, bert_lengths, bert_labels = [], [], []
            mapping = []
            for s, l in zip(sentences, labels):
                bert_label, mapping_list = [], []
                bert_sentence = TOKENIZER.convert_tokens_to_ids(["[CLS]"])
                for i, (word, word_label) in enumerate(zip(s, l)):
                    word = TOKENIZER.tokenize(word)
                    word = TOKENIZER.convert_tokens_to_ids(word)
                    bert_sentence.extend(word)
                    bert_label.extend([word_label] * len(word))
                    mapping_list.extend([i] * len(word))
                mapping.append(mapping_list)
                bert_sentences.append(
                    bert_sentence +
                    TOKENIZER.convert_tokens_to_ids(["[SEP]"]))
                bert_labels.append(bert_label)
                assert len(bert_sentence) == (len(bert_label) + 1)
                bert_lengths.append(len(bert_sentence) + 1)
            maxi = max([len(s) for s in bert_sentences])

            # Construct padded texts, compute token indices, construct mask
            tokens, mask = [], []
            for i, s in enumerate(bert_sentences):
                mask.append([1] * len(s) + [0] * (maxi - len(s)))
                tokens.append(s + [0] * (maxi - len(s)))
                bert_labels[i] = bert_labels[i] + [-2] * (maxi - len(s))
            return tokens, bert_lengths, mask, bert_labels, mapping

        sentences, metaphoricity_labels, novelty_labels = zip(*batch)
        bert_tokens, bert_lengths, bert_mask, bert_metaphoricity_labels, mapping = \
            bert_process_sentences(sentences, metaphoricity_labels)
        _, _, _, bert_novelty_labels, _ = \
            bert_process_sentences(sentences, novelty_labels)
        return Batch(
            sentences,
            bert_lengths,
            bert_mask,
            bert_tokens,
            mapping,
            bert_metaphoricity_labels,
            bert_novelty_labels,
            metaphoricity_labels,
            novelty_labels)


def load_data(batch_size=32):
    """Given a batch size load VUA and novelty scores from file.
    Args:
        batch_size: int
    Returns:
        train: DataLoader object from which you can sample batches
        dev: DataLoader object from which you can sample batches
        text: DataLoader object from which you can sample batches
    """
    novelty_scores = dict()
    with codecs.open(f"data/VUA/VUA_novelty_scores.csv", encoding="utf-8",
                     errors='ignore') as file:
        lines = csv.reader(file)
        next(lines)
        for line in lines:
            words = []
            scores = []
            for w in line[2].split():
                if "_" in w:
                    w, score = w.split("_")
                    words.append(w)
                    scores.append(float(score))
                else:
                    words.append(w)
                    scores.append(-2)
            if words:
                novelty_scores[' '.join(words)] = scores

    meta_train = get_metaphor_data(
        "data/VUA/VUA_train_small.csv", novelty_scores, train=True, batch_size=batch_size
    )
    meta_dev = get_metaphor_data(
        "data/VUA/VUA_validation.csv", novelty_scores, batch_size=batch_size)
    meta_test = get_metaphor_data(
        "data/VUA/VUA_test.csv", novelty_scores, batch_size=batch_size)
    return meta_train, meta_dev, meta_test

def load_caption_data(batch_size=32):
    novelty_scores = dict()
    with codecs.open(f"data/VUA/VUA_novelty_scores.csv", encoding="utf-8",
                     errors='ignore') as file:
        lines = csv.reader(file)
        next(lines)
        for line in lines:
            words = []
            scores = []
            for w in line[2].split():
                if "_" in w:
                    w, score = w.split("_")
                    words.append(w)
                    scores.append(float(score))
                else:
                    words.append(w)
                    scores.append(-2)
            if words:
                novelty_scores[' '.join(words)] = scores

    meta_train = get_metaphor_data(
        "data/VUA/VUA_train_small.csv", novelty_scores, train=True, batch_size=batch_size
    )
    meta_dev = get_metaphor_data(
        "data/VUA/VUA_validation.csv", novelty_scores, batch_size=batch_size)

    with open('data/captions.json', encoding='utf-8') as inputfile:
        df = pd.read_json(inputfile)
    df.to_csv('captions_test.csv', encoding='utf-8', index=False)
    meta_test = get_metaphor_data(
        "caption_test.csv", novelty_scores, batch_size=batch_size)
    return meta_train, meta_dev, meta_test


def get_metaphor_data(filename, novelty_scores, train=False, batch_size=64):
    """Prepare the DataLoader and TextDataset objects to load data with.
    Args:
        filename (str): filename of VUA metaphor dataset
        novelty_scores (dict): maps sentence to list of novelty scores
        train (bool): indicates if it's a training set
        batch_size (int): batch size
    Returns:
        DataLoader
    """
    snts, metaphoricity_labels, novelty_labels = [], [], []

    with codecs.open(filename, encoding="utf-8", errors='ignore') as filename:
        lines = csv.reader(filename)
        next(lines)
        for line in lines:
            sentence = line[2].replace("M_", "").replace("L_", "").split()
            if sentence:
                label_seq = []
                for w in line[2].split():
                    if "L_" in w:
                        label_seq.append(0)
                    elif "M_" in w:
                        label_seq.append(1)
                    else:
                        label_seq.append(-2)

                assert len(label_seq) == len(sentence)

                snts.append(sentence)
                metaphoricity_labels.append(label_seq)
                novelty_labels.append(novelty_scores[' '.join(sentence)])

    dataset = CustomDataset(
        list(snts), list(metaphoricity_labels),
        list(novelty_labels))
    # Get dataloaders to give us batches
    if train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=CustomDataset.collate_fn
    )


if __name__ == "__main__":
    load_data()