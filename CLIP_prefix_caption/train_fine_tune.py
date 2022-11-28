import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel,get_linear_schedule_with_warmup

from typing import Tuple, Optional, Union
#from datasets import load_dataset
import torch.optim as optim

#finetine gpt2 on creative vua metaphor corpus after training on coco 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Fig_Dataset(Dataset):  
    def __init__(self, data, tokenizer, max_length=23):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        tokenizer.pad_token=tokenizer.eos_token
        for i in data:
            encodings_dict = tokenizer('<BOS>' + i + '<EOS>',
                                     truncation=True,
                                     max_length=max_length,
                                     padding='max_length')

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]



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
    with open("./train_vua.csv", "r") as train:
        text = train.readlines()
    #text = text[:100]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)
    #optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    max_length = 23

    learning_rate = 1e-4
    eps = 1e-8
    warmup_steps = 50
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=eps)
    EPOCHS = 10
    
    # create text generation seed prompt
    RANDOM_SEED = 73
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # prompt = "<BOS>"
    # generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    # generated = generated.to(device)

    dataset = Fig_Dataset(text,tokenizer)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)

    #model training 
    print(len(dataloader))
    for epoch_i in range(0, EPOCHS):

        print(f'Epoch {epoch_i + 1} of {EPOCHS}')

      
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()        

            outputs = model(b_input_ids,
                                        labels=b_labels,
                                        attention_mask=b_masks,
                                        token_type_ids=None)

            loss = outputs[0]
            print(loss)  

            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

    avg_train_loss = total_train_loss / len(dataloader)       
  
    print(f'Average Training Loss: {avg_train_loss}.')

    PATH = './gpt_finetuned_weights.pt'
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


