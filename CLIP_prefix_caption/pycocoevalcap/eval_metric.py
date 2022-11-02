from pycocotools.coco import COCO
from .eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
import requests
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

annFile='/Users/aditipatil/creative_captions/CLIP_prefix_caption/data/coco/annotations/sample_captions_val2014.json'
#subtypes=['results', 'evalImgs', 'eval']
resFile = '/Users/aditipatil/creative_captions/CLIP_prefix_caption/captions_for_metrics.json'

coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()

if __name__ == '__main__':
    for metric, score in cocoEval.eval.items():
        print( '%s: %.3f'%(metric, score))