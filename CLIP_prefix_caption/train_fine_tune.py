import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel,get_linear_schedule_with_warmup

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


def main():
    #trying another way 
    with open("./train_vua.csv", "r") as train:
        text = train.readlines()
  
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)
    
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

if __name__ == '__main__':
    main()


