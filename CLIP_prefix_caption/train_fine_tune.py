import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

from typing import Tuple, Optional, Union
#from datasets import load_dataset
import torch.optim as optim

#finetine gpt2 on creative vua metaphor corpus after training on coco 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Fig_Dataset(Dataset):  
    def __init__(self, tokenizer, text, max_len):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        self.text = text
        self.result = []
        self.extra_length = len(tokenizer.encode(" TL;DR "))

        for text in self.text:
            # Encode the text using tokenizer.encode(). We add EOS at the end
            tokenized = self.tokenizer.encode(text + self.eos)
            
            # Padding/truncating the encoded sequence to max_len 
            padded = self.pad_truncate(tokenized)            

            # Creating a tensor and adding to the result
            self.result.append(torch.tensor(padded))

    def __len__(self):
        return len(self.result)


    def __getitem__(self, item):
        return self.result[item]

    def pad_truncate(self, name):
        name_length = len(name) - self.extra_length
        if name_length < self.max_len:
            difference = self.max_len - name_length
            result = name + [self.eos_id] * difference
        elif name_length > self.max_len:
            result = name[:self.max_len + 3]+[self.eos_id] 
        else:
            result = name
        return result



# def train(model, optimizer, dl, epochs):    
#     for epoch in range(epochs):
#         for idx, batch in enumerate(dl):
#              with torch.set_grad_enabled(True):
#                 optimizer.zero_grad()
#                 batch = batch.to(device)
#                 output = model(batch, labels=batch)
#                 loss = output[0]
#                 loss.backward()
#                 optimizer.step()
#                 if idx % 50 == 0:
#                     print("loss: %f, %d"%(loss, idx))

def main():
    #trying another way 
    with open("/Users/ananyasahu/nlp_project/CLIP_prefix_caption/CLIP_prefix_caption/train_vua.csv", "r") as train:
        text = train.readlines()
    text = text[:51]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    max_length = 100

    dataset = Fig_Dataset(tokenizer, text, max_length)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    #model training 
    epochs = 1 #change to 10 at least 
    dl = dataloader
    for epoch in range(epochs):
        for idx, batch in enumerate(dl):
             with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                batch = batch.to(device)
                output = model(batch, labels=batch)
                loss = output[0]
                loss.backward()
                optimizer.step()
                if idx % 50 == 0:
                    print("loss: %f, %d"%(loss, idx))

    PATH = './gpt_finetuned_weights'
    torch.save(model.state_dict(), PATH)


    #end try

    #just to see if it loads
    # dataset = load_dataset('csv', 
    # data_files={'train': "/Users/ananyasahu/nlp_project/CLIP_prefix_caption/CLIP_prefix_caption/train_vua.csv",
    # 'validation': "/Users/ananyasahu/nlp_project/CLIP_prefix_caption/CLIP_prefix_caption/val_vua.csv"},column_names=['sentence']) 


    # train_dataset = dataset['train']['sentence']
   

 
    # validation_dataset = dataset['validation']['sentence']
  
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer.pad_token = tokenizer.eos_token
    # train_encoding = tokenizer(train_dataset, padding=True, truncation=True, max_length=100, return_tensors='pt') #changed max length from 1024 to 100
    # eval_encoding = tokenizer(validation_dataset, padding=True, truncation=True, max_length=100, return_tensors='pt')
    
    # train_dataset = Dataset(train_encoding,train_dataset)
    # eval_dataset = Dataset(eval_encoding,validation_dataset)
    
    # #change for creative dataset 
    
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
  
    # training_args = TrainingArguments(
    #     output_dir= './results',
    #     num_train_epochs=3, 
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=32,
    #     warmup_steps=100, 
    #     weight_decay=0.01, 
    #     logging_dir='./logs'
    #      )
    
    # trainer = Trainer(
    #     model=model,
    #     args = training_args,
    #     train_dataset=train_encoding, #train_dataset
    #     eval_dataset=eval_encoding, #eval_dataset

    # )

    # trainer.train()
    # trainer.save_model("./gpt2finetuned")

if __name__ == '__main__':
    main()


