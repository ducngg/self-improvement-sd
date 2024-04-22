import torch_fidelity
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torch_fidelity.helpers import vassert
import random
import numpy as np
import time
import json
REF_FOLDER = "json"
image_list = []
with open("dataset.json", "r") as f:
    dataset = json.load(f)

for filename in dataset:
        image_list.append(filename)
        # print(filename)
random.shuffle(image_list)
class TransformPILtoRGBTensor:
    def __call__(self, img):
        vassert(type(img) is Image.Image, "Input is not a PIL.Image")
        img = img.resize((512,512))
        return F.pil_to_tensor(img)


class ImagesPathDataset(Dataset):
    def __init__(self, path_list, transforms=None):
        self.path_list = path_list
        self.transforms = TransformPILtoRGBTensor() if transforms is None else transforms

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, i):
        path = self.path_list[i]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return img

eval_dict = {}
eval_dict['LLaVA-13b']={}
eval_dict['LLaVA-7b']={}
eval_dict['GPT-4V']={}
eval_dict['CogVLM']={}
n_batch = 5000
for pipe in eval_dict:
    i = 1
    eval_dict[pipe]['w_verify']={}
    eval_dict[pipe]['wo_verify']={}
    eval_dict[pipe]['wo_verify']['isc']=[]
    eval_dict[pipe]['wo_verify']['fid']=[]
    eval_dict[pipe]['w_verify']['isc']=[]
    eval_dict[pipe]['w_verify']['fid']=[]
    n_loop = int(len(image_list)/n_batch)
    for i in range(n_loop):
        img_list = image_list[n_batch*i:n_batch*(i+1)]
        path_list = [os.path.join("./result", pipe, str(img_id), "1.png") for img_id in img_list]
        generator = ImagesPathDataset(path_list)
        path_list = [os.path.join("./ground_truth", f"{img_id}.png") for img_id in img_list]
        generator0 = ImagesPathDataset(path_list)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=generator,
            input2=generator0,
            cuda=True,
            isc=True,
            fid=True,
            verbose=False,
        )
        print(metrics_dict)
        eval_dict[pipe]['wo_verify']['isc'].append(metrics_dict['inception_score_mean'])
        eval_dict[pipe]['wo_verify']['fid'].append(metrics_dict['frechet_inception_distance'])
        path_list = []
        for img_id in img_list:
            path = os.path.join("./result", pipe, str(img_id), f"{img_id}.json")
            with open(path, "r") as f:
                img_step = json.load(f)
            resid = img_step['iterations']
            if resid > 5: resid = 5
            path_list.append(os.path.join("./result", pipe, str(img_id), f"{resid}.png"))
        generator = ImagesPathDataset(path_list)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=generator,
            input2=generator0,
            cuda=True,
            isc=True,
            fid=True,
            verbose=False,
        )
        print(metrics_dict)
        eval_dict[pipe]['w_verify']['isc'].append(metrics_dict['inception_score_mean'])
        eval_dict[pipe]['w_verify']['fid'].append(metrics_dict['frechet_inception_distance'])
        print(f"\r{i}/{n_loop}", end="")
        i+=1
    print("\n")
i=1
eval_dict['baseline']={}
eval_dict['baseline']['isc']=[]
eval_dict['baseline']['fid']=[]
n_loop = int(len(image_list)/n_batch)
for i in range(n_loop):
    img_list = image_list[n_batch*i:n_batch*(i+1)]
    path_list = [os.path.join("./stable_diffusion", f"{img_id}.png") for img_id in img_list]
    generator = ImagesPathDataset(path_list)
    path_list = [os.path.join("./ground_truth", f"{img_id}.png") for img_id in img_list]
    generator0 = ImagesPathDataset(path_list)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=generator,
        input2=generator0,
        cuda=True,
        isc=True,
        fid=True,
        verbose=False,
    )
    eval_dict['baseline']['isc'].append(metrics_dict['inception_score_mean'])
    eval_dict['baseline']['fid'].append(metrics_dict['frechet_inception_distance'])
    print(f"\r{i}/{n_loop}", end="")
    i+=1
with open("evalation.json","w") as f:
    json.dump(eval_dict, f)
    