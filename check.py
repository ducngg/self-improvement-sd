import argparse
from datasets import load_dataset
import json
import requests
import base64
from PIL import Image
from io import BytesIO
import time
import os
import sys
import shutil
import random
from call import *

OUTPUT_FOLDER = input("Your result folder? (eg. result/CogVLM): ")

def seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))

PERIOD = 5

print("EACH DOT • is a folder that has been done!")
    
# Open dataset metadata file
with open("dataset.json", 'r') as f:
    dataset = json.load(f)

deltas = []
times = []

i = 0
completes = 0

for image_id in dataset:

    i += 1

    if os.path.exists(os.path.join(OUTPUT_FOLDER, image_id, image_id+'.json')) and os.path.exists(os.path.join(OUTPUT_FOLDER, image_id)):
        # DONE
        completes += 1
        print("• ", end='')
    elif os.path.exists(os.path.join(OUTPUT_FOLDER, image_id)):
        # WORKING
        print((i, image_id, "Working on"))
    else: 
        # Not come yet
        pass

print(completes)
    
prev_completes = completes
time.sleep(PERIOD)

while True:
    i = 0
    completes = 0

    for image_id in dataset:

        i += 1

        if os.path.exists(os.path.join(OUTPUT_FOLDER, image_id, image_id+'.json')) and os.path.exists(os.path.join(OUTPUT_FOLDER, image_id)):
            # DONE
            completes += 1
            # print("• ", end='')
        elif os.path.exists(os.path.join(OUTPUT_FOLDER, image_id)):
            # WORKING
            print((i, image_id, "Working on"))
        else: 
            # Not come yet
            pass
    
    deltas.append(completes - prev_completes)
    times.append(PERIOD)
    try:
        # Speed = d_ID / d_t
        print(f"\t• Spd: {sum(deltas)/sum(times)*60:.2f} id/min")
        # Time = ID_left / Speed
        print(f"\t• Est: {seconds_to_hhmmss((len(dataset)-completes)/(sum(deltas)/sum(times)))}")
    except ZeroDivisionError:
        pass
    prev_completes = completes
    time.sleep(PERIOD)