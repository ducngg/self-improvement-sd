from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
import torch
from flask import Flask, redirect, request, render_template, url_for
import json
import os
import argparse
import socket
import base64
import io
from io import BytesIO
from PIL import Image
import time
import numpy as np
import sys

# model_id = "stabilityai/sdxl-turbo"
# pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
# pipe = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# ).to("cuda")

app = Flask(__name__)

# Main function to handle request
@app.route("/api", methods=['GET', 'POST'])
def handle_request():
    start_time = time.time()
    # ...
    prompt = request.form.get('prompt')
    
    # image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0] 
    image = pipe(prompt, generator=torch.manual_seed(19)).images[0]  
    image.save('sample.png')

    # image = image.resize((336, 336))

    image_io = BytesIO()
    image.save(image_io, format='PNG')
    image_binary = image_io.getvalue()
    encoded_image = base64.b64encode(image_binary).decode('utf-8')
    
    process_time = time.time() - start_time
    print(f'Done processing in {process_time:0.4f} secs')

    return {
        'image': encoded_image
    }

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 5000
    
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    app.run(host='0.0.0.0', debug=False, port=port)