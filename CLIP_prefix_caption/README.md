# Metric for Creative Image Captioning


## Implementation for the "Creative Metric for Evaluating Image Caption Generation"



## Description  
Image captioning is a complicated task, where usually a pretrained detection network is used, requires additional supervision in the form of object annotation. We present a new approach that does not requires additional information (i.e. requires only images and captions), thus can be applied to any data. In addition, our model's training time is much faster than similar methods while achieving comparable to state-of-the-art results, even for the Conceptual Captions dataset contains over 3M images. 

In our work, we use the [CLIP](https://github.com/openai/CLIP) model, which was already trained over an extremely large number of images, thus is capable of generating semantic encodings for arbitrary images without additional supervision. To produce meaningful sentences we fine-tune a pretrained language model, which has been proven to be successful for other natural language tasks. The key idea is to use the CLIP encoding as a prefix to the textual captions by employing a simple mapping network over the raw encoding, and then fine-tune our language model to generate a valid caption. In addition, we present another variant, where we utilize a transformer architecture for the mapping network and avoid the fine-tuning of GPT-2. Still, our light model achieve comaparable to state-of-the-art over nocaps dataset.

## COCO Examples

<table>
  <tr>
    <td><img src="Images/COCO_val2014_000000562207.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000165547.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000579664.jpg" ></td>
  </tr>
  <tr>
    <td>A couple of people standing next to an elephant. </td>
     <td>A wooden table sitting in front of a window.</td>
     <td>A bunch of bananas sitting on top of a table.</td>
  </tr>
 </table>
 
 <table>
  <tr>
    <td><img src="Images/COCO_val2014_000000060623.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000386164.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000354533.jpg" ></td>
  </tr>
  <tr>
    <td>A woman holding a plate with a piece of cake in front of her face. </td>
     <td>A wooden table topped with lots of wooden utensils.</td>
     <td>A red motorcycle parked on top of a dirt field.</td>
  </tr>
 </table>


Both [COCO](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing) and [Conceptual Captions](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing) pretrained models are available for mlp mapping network. For the transformer (without fine-tuning GPT-2) we provide [COCO](https://drive.google.com/file/d/1GYPToCqFREwi285wPLhuVExlz7DDUDfJ/view?usp=sharing) pretrained model.


## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook]&#40;https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing&#41; )
Clone, create environment and install dependencies:  
```
git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption
conda env create -f environment.yml
conda activate clip_prefix_caption
```
## Overview of Training and Predictions

To train the CLIP-GPT2 baseline model, we first parse the data to extract CLIP features. Then we train the model with Fine-tuning of GPT2. Finally we generate predictions. To train the CLIP-GPT2 conditioned model with the first fine-tuning method, we fine-tune GPT2 on the cleaned creative corpora provided in our repo and train GPT2 seperatley first. Then we train the CLIP-GPT2 model with fine-tuning of GPT2 with loaded GPT2 weights from our first fine-tuning. Then we predict as with the baseline. To train the CLIP-GPT2 conditioned model with thesecond fine-tuning method, we train CLIP-GPT2 model with only training the transformer mapping network. We then use the weights from our fine-tuned GPT2 on the creative copora at predicition time to generate our captions. Below are the steps and commands that correspond to each step for the training the model.
## COCO training

Download train_captions from https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing to `data/coco/annotations`.

Download training images from http://images.cocodataset.org/zips/train2014.zip and validation images http://images.cocodataset.org/zips/val2014.zip 

Extract CLIP features using (output is `data/coco/oscar_split_ViT-B_32_train.pkl`):
```
python3 parse_coco.py --clip_model_type ViT-B/32
```
Train with fine-tuning of GPT2:
```
python3 train.py --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/
```

Train only transformer mapping network:
```
python3 train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 10 --prefix_length_clip 10
```

Fine-tune GPT2 on creative corpus:
```
python3 train_fine_tune.py 
```

Generate Predictions for Fine-tuning method 1 and baseline:
```
python3 predict.py 
```

Generate Predictions for Fine-tuning method 2:
```
python3 predict_finetune2.py 
```




## Acknowledgments
This repository is heavily based on [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption) and [CLIP](https://github.com/openai/CLIP) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training we used the data of [COCO dataset](https://cocodataset.org/#home) and [VUA corpus](http://www.vismet.org/metcor/documentation/home.html).

