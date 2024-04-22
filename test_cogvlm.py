from gradio_client import Client
import time


prompt = {
    'what': 'What is in this picture?',
    'describe': 'Describe the picture',
    'details': 'Please rewrite the sentence describing the image in more detail, clarity and short in English only so that the generated image is highly detailed and sharp focus. Description: ',
    'happy': 'Describe this picture happier, with smile and laugh',
    'list': 'List the things that appeared in this picture, also list the relevant objects',
    'verify': 'Please let me know whether the image is of good quality and suitable with the description or not. Answer with only a \"Yes\" or \"No\". Description: ',
    'extend': 'Stable Diffusion is an AI art generation model similar to DALLE-2. Please write me a detailed prompt for generating art with Stable Diffusion exactly about the description follow the following rules:\
        - Prompt should always be written in English, regardless of the input language. Please provide the prompt in English.\
        - Prompt should consist of a description of the scene followed by modifiers divided by commas.\
        - When generating description, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.\
        - The modifiers should alter the mood, style, lighting, and other aspects of the scene.\
        - Multiple modifiers can be used to provide more specific details.\
        Please write only exactly a prompt.\
        Description: ',
    
    'limited_details': 'Please improve the description image more highly detailed and shortly in English. Description: ',
    'verify_2': 'Please let me know if the image is good quality and suitable with the description or not, answer Yes if so.',
    'feedback': 'If no, what things the description has but the image doesn\'t have, answer shortly. Description: '
} 

start_time = time.time()

client = Client("https://e43763510c939e3939.gradio.live")
result = client.predict(
		prompt['details'] + "A toy car that has 4 wheels" + ". Just do what I say, do not care about the image",	# str  in 'Input Text' Textbox component
		0.9,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
		0.7,	# float (numeric value between 0 and 1) in 'Top P' Slider component
		5,	# float (numeric value between 1 and 100) in 'Top K' Slider component
		"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# filepath  in 'Image Prompt' Image component
		[],	# [[Question, Answer]]
		"",	# str  in 'parameter_22' Textbox component
		api_name="/post"
)
# print(result)
print(result[1][0][1])

print(time.time() - start_time)


start_time = time.time()

result = client.predict(
		prompt['verify'] + "A toy car that has 4 wheels",	# str  in 'Input Text' Textbox component
		0.2,	# float (numeric value between 0 and 1) in 'Temperature' Slider component
		0.5,	# float (numeric value between 0 and 1) in 'Top P' Slider component
		2,	# float (numeric value between 1 and 100) in 'Top K' Slider component
		"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# filepath  in 'Image Prompt' Image component
		[["Hello, please answer the below question, command as short as possible", "I will response as short as possible, but still giving you the best answer"]],	# [[Question, Answer]]
		"",	# str  in 'parameter_22' Textbox component
		api_name="/post"
)
# print(result)
print(result)

print(time.time() - start_time)




