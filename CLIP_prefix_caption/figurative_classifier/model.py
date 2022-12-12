"""
Neural network model for metaphor prediction.
- BERT:
    - BERT-base-cased
    - a classification layer for metaphoricity with sigmoid
    - a classification layer for novelty with tanh
"""
import torch

from transformers import BertModel


class MetaphorModel(torch.nn.Module):
    """Metaphor detection model."""

    def __init__(self):
        super().__init__()
        self.name = f"model=bert"

        #if self.bert:
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.train()

        # Simple classifier
        self.dropout_output = torch.nn.Dropout(0.1)
        self.metaphor_output_projection = torch.nn.Linear(768, 1)
        self.novelty_output_projection = torch.nn.Linear(768, 1)

    def forward(self, inputs, mask):
        """
        Forward pass of BERT model for metaphor prediction.
        Args:
            inputs: list of lists, containing tokenised sequences of strings
            mask: PyTorch LongTensor of zeros and ones
        Returns:
            metaphoricity: batch_size x max_sent_length of float predictions
            novelty: batch_size x max_sent_length of float predictions
        """
        # Extract word encodings from BERT
        if not torch.cuda.is_available():
            bert_output = self.bert_model(
                input_ids=inputs, attention_mask=mask)[0]
        else:
            bert_output = self.bert_model(
                input_ids=inputs.cuda(), attention_mask=mask.cuda())[0]
        encoded_sentence = self.dropout_output(bert_output)
        metaphoricity = torch.sigmoid(
            self.metaphor_output_projection(encoded_sentence)).squeeze(-1)
        novelty = torch.tanh(
            self.novelty_output_projection(encoded_sentence)).squeeze(-1)
        # Don't pass the CLS and SEP tokens
        return metaphoricity[:, 1:-1], novelty[:, 1:-1]