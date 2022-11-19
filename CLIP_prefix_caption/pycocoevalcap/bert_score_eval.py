from torchmetrics.text.bert import BERTScore
import statistics
import json

annFile='/Users/ananyasahu/nlp_project/CLIP_prefix_caption/CLIP_prefix_caption/data/coco/annotations/train_caption_filtered.json'
resFile = '/Users/ananyasahu/nlp_project/CLIP_prefix_caption/CLIP_prefix_caption/captions_for_metric.json'

with open(annFile) as json_file:
    data_ref = json.load(json_file)

with open(resFile) as json_file:
    data_pred = json.load(json_file)

preds = []
refs = []
for pred_dict in data_pred:
    for ref_dict in data_ref:
        if pred_dict["image_id"] == ref_dict["image_id"]:
            preds.append(pred_dict["caption"])
            refs.append(ref_dict["caption"])


bertscore = BERTScore()
score = bertscore(preds, refs)
rounded_score = {k: [round(v, 3) for v in vv] for k, vv in score.items()}

print(statistics.mean(rounded_score['f1']))
print(statistics.mean(rounded_score['precision']))
print(statistics.mean(rounded_score['recall']))