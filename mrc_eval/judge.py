import json
import os
import openai
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path,"config.json")) as f:
    data = json.load(f)

openai.organization = data["organization"]
openai.api_key = data["api_key"]