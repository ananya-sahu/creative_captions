# import skimage.io as io
# import torch
# import clip
# from PIL import Image
# import pickle
# import json
# import os
# from tqdm import tqdm
# import argparse


# def main(clip_model_type: str):
#     #device = torch.device('cuda:0') 
#     device = torch.device('cpu')
#     clip_model_name = clip_model_type.replace('/', '_')
#     out_path = f"./data/coco/oscar_split_{clip_model_name}_train.pkl"
#     clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
#     with open('./data/coco/annotations/train_caption.json', 'r') as f:
#         data = json.load(f)
#     print("%0d captions loaded from json " % len(data))
#     all_embeddings = []
#     all_captions = []
#     half = len(data)
#     count = 0
#     for i in tqdm(range(1000)): #changed len(data) to 100
#         #changed from i to count
#         d = data[count]
#         img_id = d["image_id"]
#         filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
#         if not os.path.isfile(filename):
#             filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
#         if os.path.isfile(filename):
#             image = io.imread(filename)
#             image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 prefix = clip_model.encode_image(image).cpu()
#             #changed here to i ->count
#             d["clip_embedding"] = count
#             all_embeddings.append(prefix)
#             all_captions.append(d)
#             #changed i -> count
#             if (count + 1) % 10000 == 0:
#                 with open(out_path, 'wb') as f:
#                     print(torch.cat(all_embeddings, dim=0))
#                     pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
#             count += 1

#     with open(out_path, 'wb') as f:
#         pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

#     print('Done')
#     print("%0d embeddings saved " % len(all_embeddings))
#     return 0


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
#     args = parser.parse_args()
#     exit(main(args.clip_model_type))
import skimage.io as io
import torch
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./data/coco/annotations/train_caption_filtered.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(500)): #len(data))
        d = data[i]
        img_id = d["image_id"]
        filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
