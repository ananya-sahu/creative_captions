import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

from typing import Tuple, Optional, Union
from datasets import load_dataset

#finetine gpt2 on creative vua metaphor corpus after training on coco 



class torchDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.len = len(encodings)

    def __getitem__(self, index):
        item = {torch.tensor(val[index]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.len

    def print(self):
        print(self.encodings)

def main():
    #just to see if it loads
    dataset = load_dataset('csv', data_files={'train': "/Users/ananyasahu/nlp_project/CLIP_prefix_caption/CLIP_prefix_caption/VUA_formatted_train.csv",'validation': "/Users/ananyasahu/nlp_project/CLIP_prefix_caption/CLIP_prefix_caption/VUA_validation.csv"}) 
    print(dataset)

    train_dataset = dataset['train']['sentence']
    validation_dataset = dataset['validation']['sentence_txt']
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_encoding = tokenizer(train_dataset, padding=True, truncation=True, max_length=1024, return_tensors='pt')
    eval_encoding = tokenizer(validation_dataset, padding=True, truncation=True, max_length=1024, return_tensors='pt')
    
    #change for creative dataset 
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    training_args = TrainingArguments(
        output_dir= './results',
        num_train_epochs=3, 
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100, 
        weight_decay=0.01, 
        logging_dir='./logs'
         )
    
    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=train_encoding, #changed from test_dataset
        eval_dataset=eval_encoding, #changed from test_eval

    )

    trainer.train()
    trainer.save_model("./gpt2finetuned")

if __name__ == '__main__':
    main()


