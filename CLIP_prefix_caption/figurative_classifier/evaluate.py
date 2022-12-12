"""Evaluation functions for the metaphor & novelty project."""

import logging
import scipy.stats
import torch
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error


def evaluate(model, dataloader, dataset_name="validation", output_filename=None,
             no_labels=False):
    """
    Evaluation metaphor model on VU Amsterdam metaphor test data.
    Args:
        model (nn.Module): model to evaluate
        dataloader (DataLoader): contains the validation batches
        dataset_name: validation / test
        output_filename (string): filename to write predictions to
    """
    model.eval()

    pairs, trace = [], []
    pairs_novelty = []

    for batch in dataloader:
        inputs = batch.tokens
        metaphoricity_outputs, novelty_outputs = \
            model(inputs, batch.mask)

        # Run model in inference mode
        for i, (prediction_m, prediction_n, length) in enumerate(
            zip(metaphoricity_outputs, novelty_outputs, batch.lengths)):
            prediction_m = torch.round(prediction_m[:length - 1].cpu().squeeze(-1)).tolist()
            prediction_m = transform(prediction_m, batch.mapping[i])
            target_m = batch.metaphoricity_labels[i]
            pairs.extend([(t, p) for t, p in \
                zip(target_m, prediction_m) if t != -2])

            prediction_n = prediction_n[:length - 1].cpu().tolist()
            prediction_n = transform(prediction_n, batch.mapping[i])
            target_n = batch.novelty_labels[i]
            pairs_novelty.extend([(t, p) for t, p in \
                zip(target_n, prediction_n) if t != -2])

            # Save a trace for analysis purposes
            trace.append((
                batch.sentences[i], target_m, prediction_m, target_n, prediction_n)
            )

    tgt, prd = zip(*pairs)
    p, r, f1, _ = precision_recall_fscore_support(tgt, prd, average="binary")
    str_ ='Validating' if dataset_name == 'validation' else 'Testing'
    logging.info(f"{str_}, Metaphoricity, P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}")

    tgt, prd = zip(*pairs_novelty)
    r = scipy.stats.pearsonr(tgt, prd)[0]
    mae = mean_absolute_error(tgt, prd)
    logging.info(f"{str_}, Novelty: Pearson's r {r}, MAE: {mae}")

    # If the output filename of the trace is specified, save the traces
    # to file
    if output_filename is not None:
        with open(output_filename, 'w', encoding="utf-8") as f:
            if no_labels:
                f.write(f"sentences\tprediction_metaphoricity\tprediction_novelty\n")
            else:
                f.write(f"sentences\ttarget_metaphoricity\tprediction_metaphoricity\ttarget_novelty\tprediction_novelty\n")                
            for sentence, target_m, prediction_m, target_n, prediction_n in trace:
                t_m = " ".join([str(x) for x in target_m])
                p_m = " ".join([str(x) for x in prediction_m])
                t_n = " ".join([str(round(x, 3)) for x in target_n])
                p_n = " ".join([str(round(x, 3)) for x in prediction_n])
                if no_labels:
                    f.write(f"{' '.join(sentence)}\t{p_m}\t{p_n}\n")
                else:
                    f.write(f"{' '.join(sentence)}\t{t_m}\t{p_m}\t{t_n}\t{p_n}\n")
    return f1, r


def transform(prediction_sent, ids_sent):
    """
    Map the BERT predictions per sentence piece back to the original words.
    Args:
        prediction_sent (list): prediction per sentence piece
        ids_sent (list): index of sentence pieces that map back to words
    Returns:
        list containing predictions for the original words
    """
    mapping = {i: None for i in range(max(ids_sent) + 1)}
    for prediction_word, index in zip(prediction_sent, ids_sent):
        if mapping[index] is None:
            mapping[index] = prediction_word
    return [mapping[i] for i in range(max(ids_sent) + 1)]