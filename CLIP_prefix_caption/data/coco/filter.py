# importing required packages

# Python program to read
# json file
  
import json
from pathlib import Path
import shutil
import os
 
# defining source and destination
# paths
f = open('annotations/train_caption.json')
data = json.load(f)


src2 = 'val2014_shortened/'
src1 = 'train2014_shortened/'


files_t=os.listdir(src1)

# # iterating over all the files in
# # the source directory
training = []
for i in range(len(files_t)):
    # copying the files to the
    # destination directory
    str1 = files_t[i].replace('.jpg', '').replace('COCO_train2014_', '') .lstrip("0")
    training.append(str1)





files_v=os.listdir(src2)

# # iterating over all the files in
# # the source directory
val = []
for i in range(len(files_v)):
    # copying the files to the
    # destination directory
    str1 = files_v[i].replace('.jpg', '').replace('COCO_val2014_', '') .lstrip("0")
    val.append(str1)






new_data = []
for d in data:
    if d['image_id'] in training:
        new_data.append(d)
    elif d['image_id'] in val:
        new_data.append(d)

with open("train_caption_filtered", "w") as outfile:
            json.dump(new_data, outfile)

print(len(data))
print(len(new_data))