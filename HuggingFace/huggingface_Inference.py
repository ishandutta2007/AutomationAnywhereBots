#Author:Prasad Katankot
#Partner Solution Desk
#May-16-2024 - Working Copy of Chat completion

from huggingface_hub import login
import os
import json
import logging
import datetime
import traceback
#os.environ['HF_TOKEN'] = "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"
#login(token="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")
from huggingface_hub import InferenceClient

# client.post
# image = client.text_to_image("An astronaut riding a horse on the moon.")
# image.save("astronaut.png")

# class1 = client.token_classification("I am not happy with the service")
# print(class1[0])

import platform
import sys
import os


def get_runtime_environment():
	environment_info = {
		"Python Version": sys.version,
		"Platform": platform.system(),
		"Platform Release": platform.release(),
		"Platform Version": platform.version(),
		"Architecture": platform.architecture(),
		"Machine": platform.machine(),
		"Processor": platform.processor(),
		"Python Implementation": platform.python_implementation(),
		"Python Build": platform.python_build(),
		"Python Compiler": platform.python_compiler(),
		"Environment Variables": dict(os.environ),
	}

	return environment_info


def print_environment_info(environment_info, logger):
	for key, value in environment_info.items():
		if key == "Environment Variables":
			print(f"{key}:")
			logger.info(f"{key}:")
			for var, val in value.items():
				print(f"  {var}: {val}")
				logger.info(f"  {var}: {val}")
		else:
			print(f"{key}: {value}")
			logger.info(f"{key}: {value}")

def logConfig(path):
#os.chdir("../../../ProgramData/AA_Python_Execution_Logs")
    if path != "":
        os.chdir(path)
    currDate = str(datetime.date.today())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    file_handler = logging.FileHandler(currDate+".log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def modelPost(inputText):
	client = InferenceClient("microsoft/Phi-3-mini-4k-instruct")

	res = client.post(json={
		"inputs": inputText,
	})
	encoding = 'utf-8'
	res = str(res, encoding)
	res = json.loads(res)
	#print(res[0]['generated_text'])
	return res[0]['generated_text']

def text_generation(args_string):

	args = args_string  # dictionary passed from the bot is available as a dict here
	logPath = args['logPath']
	token = args['token']
	login(token=token)
	logger = logConfig(logPath)
	# environment_info = get_runtime_environment()
	# print_environment_info(environment_info, logger)
	logger.info('message from main module')
	logger.info(args_string)
	logger.info(logPath)
	model = args['model']
	logger.info('model-' + model)
	prompt = args['prompt']
	max_tokens = int(args['max_tokens'])
	messages = [{"role": "user", "content": prompt}]
	if model !='':
		client = InferenceClient(model)
	else:
		client = InferenceClient()
	text_gen = client.text_generation(prompt, max_new_tokens=max_tokens)
	#print(text_gen)
	return text_gen


def chat(args_string):
	try:
		#logger = logConfig("../../../ProgramData/AA_Python_Execution_Logs") pass this same path or alternative path from the bot
		args=args_string # dictionary passed from the bot is available as a dict here
		logPath = args['logPath']
		token=args['token']
		login(token=token)
		logger = logConfig(logPath)
		# environment_info = get_runtime_environment()
		# print_environment_info(environment_info, logger)
		# logger.info('message from main module')
		logger.info(args_string)
		# logger.info(logPath)
		model= args['model']
		logger.info('model-'+model)
		prompt= args['prompt']
		max_tokens= int(args['max_tokens'])
		messages = [{"role": "user", "content":prompt }]
		if model != '':
			client = InferenceClient(model)
		else:
			client = InferenceClient()
		chat_completion=client.chat_completion(messages, max_tokens=max_tokens)
		DetailsDict = {"status": "Success","details": chat_completion.choices[0].message.content}
		Details = json.dumps(DetailsDict)
		return Details
	except Exception as e:
		line_number_1 = traceback.extract_tb(e.__traceback__)[-1].lineno
		# Get the error message
		error_message_1 = str(e)
		logger.error(f"line_number: {line_number_1}, error_message:{error_message_1}")
		errDetailsDict= {"status":"error", "details":f"line_number: {line_number_1} error_message:{error_message_1} in Python Code"}
		errDetails=json.dumps(errDetailsDict)
		return errDetails

if __name__ == "__main__":
	#print(chat("HuggingFaceH4/zephyr-7b-beta", "What is the capital of France?",100))
	strval={'max_tokens': '100', 'model': 'HuggingFaceH4/zephyr-7b-beta', 'prompt': 'What is the capital of France?','logPath':'','token':'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'}
	#strval = strval.replace("\'", "\"")
	print(chat(strval))
	# environment_info = get_runtime_environment()
	# print_environment_info(environment_info)
	#print(modelPost("Which city is capital of India"))




