#!pip install torch==1.5.1
#!pip install transformers==3.0.1
#!pip install bert_score==0.3.4

import statistics
import json
from bert_score import score

annFile = '/content/train_caption_filtered.json'
resFile = '/content/captions_for_metric.json'

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



P, R, F1 = score(preds, refs, lang="en", verbose=True)

print(f"System level P score: {P.mean():.3f}")
print(f"System level R score: {R.mean():.3f}")
print(f"System level F1 score: {F1.mean():.3f}")