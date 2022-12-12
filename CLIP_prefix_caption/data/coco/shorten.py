# importing required packages
from pathlib import Path
import shutil
import os
 
# defining source and destination
# paths
src = '/Users/ananyasahu/nlp_project/CLIP_prefix_caption/CLIP_prefix_caption/data/coco/val2014/'
trg = '/Users/ananyasahu/nlp_project/CLIP_prefix_caption/CLIP_prefix_caption/data/coco/val2014_shortened/'
 
files=os.listdir(src)
shortened = 5000

 
# # iterating over all the files in
# # the source directory
for i in range(shortened):
     
    # copying the files to the
    # destination directory
    shutil.copy(src+files[i], trg)