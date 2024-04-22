'''
python main.py --enhancer=... --validator=... --resume=True|False --set=... --sd=...
'''
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

MAX_ITER = 5

prompt = {
    'details': 'Please rewrite the sentence describing the image in more detail, clarity and short in English only so that the generated image is highly detailed and sharp focus. Description: ',
    'verify': 'Please let me know whether the image is of good quality and suitable with the description or not. Remember answer with only a \"Yes\" or \"No\". Description: '
}

def seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))

def saveBase64(base64str, path="sample.png"):
    decoded_data = base64.b64decode(base64str)
    image_stream = BytesIO(decoded_data)
    image = Image.open(image_stream)
    image.save(path)
    print(f"\t-> Image saved as {path} {image.size}")

def readImageFromUrl(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_data = response.content
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        return base64_encoded
    else:
        print(f"Failed to fetch the image. Status code: {response.status_code}")
        return None

def accept(response):
    yeskeywords = ['yes', 'Yes', 'it is suitable', 'It is suitable', 'is suitable']
    return any(keyword in response for keyword in yeskeywords)

def create(args, input, enhancer, validator, sd_url, target=prompt['details'], max_iter=10, id='sample'):
    id = str(id)
    cost = []
    generating_process = []
    i = 1
    input0 = input
    
    gpt_respone = ''
    folderPath = os.path.join(OUTPUT_FOLDER, id)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    
    # Set verifying prompt to ask gpt if the generated image suits with input
    # verifying_prompt = prompt['verify_2'] + prompt['feedback'] + '\"' + input + '\"'
    verifying_prompt = prompt['verify'] + '\"' + input + '\"'
    rewrite_prompt = prompt['details'] + '\"' + input + '\"'
    
    while i <= max_iter:

        print(f"\nIter: {i}")
        
        # Improve input_0 using enhancer
        while True:
            try:
                sd_prompt, process_time = enhancer(rewrite_prompt)
            except Exception as e:
                print(f"Failed to make the request. Error: {e}")
                print("\t\tFAILED")
                time.sleep(5)
                continue
            break
            
        cost.append(process_time)
        
        # Generate image for iteration i
        print(f"\t[Generating image] ({sd_prompt[:50]}...) ...")
        image, process_time = getImageFromSD(sd_prompt, sd_url)
        cost.append(process_time)
        print(f"\t[Generating image][DONE] in {process_time:0.1f} secs:")
        savePath = os.path.join(folderPath, f"{i}.png")
        saveBase64(image, path=savePath)
        
        # CogVLM API uses path to image as imput
        if args.validator == 'cogvlm':
            image = savePath
            
            
        # Verify generated image using verifier
        while True:
            try:
                response, process_time = validator(verifying_prompt, image)
            except Exception as e:
                print(f"Failed to make the request. Error: {e}")
                print("\t\tFAILED")
                time.sleep(5)
                continue
            break
        
        cost.append(process_time)

        generating_process.append((i, sd_prompt, response))
        
        if accept(response):
            break
            
        i+=1
        
    
    if i > max_iter:
        i = max_iter
    
    ret = {
        'iterations': i,
        'input': input,
        'cost': sum(cost),
        'generating_process': generating_process
    }
    
    # Save json file
    with open(os.path.join(folderPath, f'{id}.json'), 'w') as jsonfile:
        json.dump(ret, jsonfile, indent=4)
    return ret

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Pipeline")

    parser.add_argument("--enhancer", type=str, help="Agent for improving image description prompts", choices=['gpt4', 'cogvlm', 'llava-13b', 'llava-7b'], required=True)
    parser.add_argument("--validator", type=str, help="Agent for verifying if generated images is suitable with original prompt", choices=['gpt4v', 'cogvlm', 'llava-13b', 'llava-7b'], required=True)
    parser.add_argument("--resume", type=str, help="Resume? True or False", default='True')
    parser.add_argument("--set", type=str, help="Portion of dataset to be executed (for parallel executing)", choices=['odd', 'even', '0mod3', '1mod3', '2mod3', 'all'], default='all')
    parser.add_argument("--sd", type=int, help="Port of the stable diffusion local server", default=5000)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Maps improving prompt agent
    if args.enhancer == 'gpt4':
        improveAgent = getResponeFromGPT4
    elif args.enhancer == 'cogvlm':
        improveAgent = getResponeFromCOGVLM
    elif args.enhancer == 'llava-13b':
        improveAgent = getResponeFromLLaVA13b
    elif args.enhancer == 'llava-7b':
        improveAgent = getResponeFromLLaVA7b
    else:
        improveAgent = None
    print(f"|| Prompt enhancer: {args.enhancer}")
    
    # Maps verifing agent
    if args.validator == 'gpt4v':
        verifyAgent = getResponeFromGPT4V
    elif args.validator == 'cogvlm':
        verifyAgent = getResponeFromCOGVLM
    elif args.validator == 'llava-13b':
        verifyAgent = getResponeFromLLaVA13b
    elif args.validator == 'llava-7b':
        verifyAgent = getResponeFromLLaVA7b
    else:
        verifyAgent = None
    print(f"|| Image validator: {args.validator}")

    # Maps result data folder
    if args.enhancer == 'gpt4' and args.validator == 'gpt4v':
        OUTPUT_FOLDER = 'result/GPT-4V'
    elif args.enhancer == 'llava-13b' and args.validator == 'llava-13b':
        OUTPUT_FOLDER = 'result/LLaVA-13b'
    elif args.enhancer == 'llava-7b' and args.validator == 'llava-7b':
        OUTPUT_FOLDER = 'result/LLaVA-7b'
    elif args.enhancer == 'cogvlm' and args.validator == 'cogvlm':
        OUTPUT_FOLDER = 'result/CogVLM'
    else:
        OUTPUT_FOLDER = 'result/default'
    print(f"|| Output folder: {OUTPUT_FOLDER}")
    
    # Open dataset metadata file
    with open("dataset.json", 'r') as f:
        dataset = json.load(f)
    # Maps the to be executed set (for parallel processing) 
    if args.set == 'odd':
        working_set = {key: value for key, value in dataset.items() if int(key) % 2 != 0}
    elif args.set == 'even':
        working_set = {key: value for key, value in dataset.items() if int(key) % 2 == 0}
    elif args.set == '0mod3':
        working_set = {key: value for key, value in dataset.items() if int(key) % 3 == 0}
    elif args.set == '1mod3':
        working_set = {key: value for key, value in dataset.items() if int(key) % 3 == 1}
    elif args.set == '2mod3':
        working_set = {key: value for key, value in dataset.items() if int(key) % 3 == 2}
    else:
        working_set = dataset
    keys = list(working_set.keys())
    print(f"|| Working set: {args.set} ({keys[:4]}...{keys[-4:]})")
    
    # Maps stable diffusion local server port
    SD_URL = f'http://127.0.0.1:{args.sd}/api'
    print(f"|| Stable diffusion server used: {SD_URL}")
        
    costs = []
    iters = []
    checkpoint = 1

    # Resume
    if args.resume == 'True':
        checkpoint = len(os.listdir(OUTPUT_FOLDER))
        if checkpoint == 0:
            checkpoint = 1
            
            
        for record in os.listdir(OUTPUT_FOLDER):
            json_path = os.path.join(OUTPUT_FOLDER, record, record+'.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as jf:
                    data = jf.read()
                    data = json.loads(data)
                    iter = data['iterations']
                    cost = data['cost']
                    
                    if iter > MAX_ITER:
                        iter = MAX_ITER
                    
                    iters.append(iter)
                    costs.append(cost)
                
        print(f"Resumed from {checkpoint}: ")
        print(f"\t Average iteration: {sum(iters)/checkpoint:>30}")
        print(f"\t Total cost:        {sum(costs):>30}")

    i = 0
    total_image = len(dataset)
    
    for image_id in dataset:
        
        i += 1
        
        if (image_id not in keys) or (i < checkpoint and os.path.exists(os.path.join(OUTPUT_FOLDER, image_id, image_id+'.json'))) or (os.path.exists(os.path.join(OUTPUT_FOLDER, image_id, image_id+'.json'))):
            continue
        
        print((i, image_id, args.set))
                
        input0 = dataset[image_id]

        print(f'\t•[{i}] Description: ', input0)
    
        result = create(args, input0, improveAgent, verifyAgent, SD_URL, target=prompt['details'], max_iter=MAX_ITER, id=image_id)
        
        costs.append(result['cost'])
        iters.append(result['iterations'])

        print(f"\t• Total cost: {sum(costs):0.1f} secs")
        print(f"\t• Average cost: {sum(costs)/len(costs):0.1f} secs")
        print(f"\t• Average iterations: {sum(iters)/len(iters):0.1f} iters")
        print(f"\t• Est: {seconds_to_hhmmss(sum(costs)/len(costs)*(total_image-i))}")
        print('==================================================\n')
        
