# Text-to-Image Generation Repository

This repository contains scripts and files for enhancing text-to-image generation using various models and techniques.

## Contents

1. `stable_diffusion.py`: This script creates a local Flask server for an API to utilize the stable diffusion model. It receives a prompt to generate an image and returns the image encoded in base64 format.

2. `call.py`: This script contains a collection of functions used to call various models, including LLaVA, stable diffusion, GPT-4, and GPT-4V, for multiple purposes related to text-to-image generation.

3. `dataset.json`: This file is a collection of descriptions about the images in the dataset. These descriptions are used as prompts for generating images using the text-to-image pipeline.

4. `main.py`: This script runs our text-to-image generation pipeline. It processes each description from `dataset.json` to enhance the quality of the generated images. Instead of inputting simple prompts directly to the stable diffusion model, this script employs various techniques to generate higher-quality images.

## Usage

1. **Stable Diffusion Server**:
   - Start the local Flask server for the stable diffusion model API by running:
     ```
     python stable_diffusion.py [PORT]
     ```
     Replace `[PORT]` with the desired port number (default is 5000).
   - Send prompts to this server to generate images and receive the encoded images in base64 format.
   
2. **Pipeline Execution**:
   - Execute the text-to-image generation pipeline by running the following command:
     ```
     python3 main.py --enhancer=AGENT --validator=AGENT
     ```
     Replace `AGENT` with the desired model for improving prompts or verifying images.
   
   - **Optional Arguments**:
     - `--resume`: Specify `True` or `False` to resume the process at the last image ID if it encounters an error.
     - `--set`: Choose from `all`, `odd`, `even`, `0mod3`, `1mod3`, `2mod3` to select a portion of the dataset to be executed. For example, `--set=odd` will run the process on images with odd IDs.
     - `--sd`: Specify the port of the local stable diffusion server using an integer, normally 5000.
   Example: 
   - Running all images dataset on one process
     ```
     python3 main.py --enhancer=gpt4 --validator=gpt4v
     python3 main.py --enhancer=cogvlm --validator=cogvlm --resume=True
     ```
   - Running parallel on 2 processes (requires 2 processes of stable diffusion server)
     ```
     python3 main.py --enhancer=llava-7b --validator=llava-7b --resume=False --set=even --sd=5000
     python3 main.py --enhancer=llava-7b --validator=llava-7b --resume=False --set=odd --sd=5001
     ```
   - Running parallel on 3 processes (requires 3 processes of stable diffusion server)
     ```
     python3 main.py --enhancer=llava-13b --validator=llava-13b --resume=True --set=0mod3 --sd=5000
     python3 main.py --enhancer=llava-13b --validator=llava-13b --resume=True --set=1mod3 --sd=5001
     python3 main.py --enhancer=llava-13b --validator=llava-13b --resume=True --set=2mod3 --sd=5002
     ```
     
## Dependencies
See `requirements.txt`.

## License

This project is licensed under the (LICENSE).
