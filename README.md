# Metric for Creative Image Captioning


## Implementation for the "Creative Metric for Evaluating Image Caption Generation"


## Environment Set Up

Cloning and set up:  
```
git clone https://github.com/ananya-sahu/creative_captions.git
conda env create -f environment.yml
conda activate clip_prefix_caption
```
## Overview of Training and Predictions

To train the CLIP-GPT2 baseline model, we first parse the data to extract CLIP features. Then we train the CLIP-GPT2 baseline model with Fine-tuning of GPT2. Finally we generate predictions. To train the CLIP-GPT2 conditioned model with the first fine-tuning method, we fine-tune GPT2 on the cleaned creative corpora (train_vua.csv) provided in our repo and train GPT2 seperatley first. Then we train the CLIP-GPT2 model with fine-tuning of GPT2 with loading the GPT2 weights from our first fine-tuning on the creative corpora. Then we predict as with the baseline. To train the CLIP-GPT2 conditioned model with the second fine-tuning method, we train CLIP-GPT2 model with only training the transformer mapping network. We then use the weights from our fine-tuned GPT2 on the creative copora at predicition time to generate our captions. Below are the steps and commands that correspond to each part for the training the model.

## COCO training

Download train_captions from https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing to `data/coco/annotations`.

Download training images from http://images.cocodataset.org/zips/train2014.zip and validation images http://images.cocodataset.org/zips/val2014.zip 

To shorten dataset cd into the data/coco folder and run:
```
python3 shorten.py 
```
followed by
```
python3 filter.py 
```


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

## Evaluation - Baseline Metrics

For baseline metrics, our code is based upon [Microsoft COCO Caption Evaluation](https://github.com/daqingliu/coco-caption) repository. The code for calcaulting and scoring the metrics must be cloned from this repository in order to evaluate them here by doing so:

```
git clone -b python3 https://github.com/XgDuan/coco-caption.git
```

We used the BLEU, CIDEr, and ROUGE folders from this repository to calculate the metric. Then, in order to evaluate on the creatively generated captions, cd into the ClIP_prefix_caption directory and run 

```
python3 -m pycocoevalcap.eval
```

To generate BertScores first copy the path of the annotated captions file and the generated captions to bert_score_eval.py.
Then run:
```
python3 bert_score_eval.py 
```

## Evaluation - Automatic Metric
To run the creative evaluation scripts, cd into the pycocoevalcap directory and upload the generated caption json files. Copy the path of the captions files into creative.py file for evaluation on all the captions. To evaluate single captions copy the caption to be evaluated into creative_scorer_indivudal.py. Note for indivudal captions, the corpus from which the caption came is still needed in order to generate a score. 

Generate scores on all captions:
```
python3 creative.py 
```

Generate scores on a single caption:
```
python3 creative_scorer_individual.py 
```


## Authored
Aditi authored the baseline metrics, except BertScore, and automatic metric's creative files. Both Aditi and Ananya authored the train and predict changes. Ananya authored the Bert Score, adjective POS tagging, train_fine_tune and predict_fine_tune code for the two fine-tuned models. Both Ananya and Aditi authored the classifier_for_weights Jupyter notebook. Also included is train_clip_bart and predict_clip_bart which was not included in the final paper, but authored by Aditi.  


## Acknowledgments
This repository is heavily based on [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption) and [CLIP](https://github.com/openai/CLIP) and [Hugging-faces](https://github.com/huggingface/transformers) repositories. The code in the files "train.py", "parse_coco.py" come directly from the ClipCap repository. "fine_tuned_train.py" is also the same as "train.py" except for a minor change of loading the new weights. 
For training we used the data of [COCO dataset](https://cocodataset.org/#home) and [VUA corpus](http://www.vismet.org/metcor/documentation/home.html).

