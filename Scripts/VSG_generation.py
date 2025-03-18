### importing libraries
import pathlib
import textwrap
from pprint import pprint
import google.generativeai as genai
import json 
from tqdm import tqdm
import os 
import numpy as np
import time


### set API keys and model
google_api_key = "ENTER_YOUR_API_KEY_HERE"
os.environ["GOOGLE_API_KEY"] = google_api_key

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')


## set safety settting
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


### function to create prompt for model
def make_prompt(sample):
    prompt = f"""Analyze the input image and its caption to identify all the objects, their attributes, and relationships between them.

    1. Identify Objects: Label each object with a specific name (e.g., 'red apple' instead of 'fruit', 'fred' or 'wilma' or 'dino' instead of man, woman, animal).
    2. Define Attributes: For each identified object, list attributes such as color, size, position in the format:
    - Object (attr1, attr2, attr3, ...)
    3. Specify Relationships: Describe relationships between objects in the format:
    - relationship(obj1, obj2)
    4. The characters depicted in the image are restricted to the following list: ['fred', 'wilma', 'barney', 'betty', 'pebble', 'slate', 'dino', 'bamm-bamm'].

    Caption : {sample['description']}

    Return a single object if multiple similar objects are detected.

    Return the result in the following JSON format without any other content:

    {{
        "objects": [
            {{"name": "object1", "attributes": ["attr1", "attr2", ...]}},
            {{"name": "object2", "attributes": ["attr1", "attr2", ...]}}
        ],
        
        "relationships": [
            "relationship(object1, object2)",
            "relationship(object3, object4)"
        ]
    }}"""

    ### set image data for Gemini
    diagram_image = {
    'mime_type': 'image/png',
    'data': pathlib.Path(os.path.join('ENTER_IMAGE_FOLDER_PATH', sample['image_id'] + '.png')).read_bytes()
    }
    return [diagram_image, prompt]


### function to generate VSG from Gemini
def generate_solution(sample):
    response = model.generate_content(
    contents = make_prompt(sample),
    generation_config = genai.GenerationConfig(
        candidate_count = 1,
        max_output_tokens = 1024,
        temperature = 1,
        top_p = 0.8,
        top_k = 20), safety_settings=safety_settings
        )

    for candidate in response.candidates:
        generated_vsg = " ".join([part.text for part in candidate.content.parts])
        return generated_vsg


### loadf the input flintstonesSV data file
with open('path/to/flintstones/data/file.json') as f:
    data = json.load(f)


### loop to generate VSG and store it in jsonl file
with open(f"path/to/data/output/file.jsonl", "a", encoding='utf-8') as ans_file:
    for sample in tqdm(data):
        img = sample['image_id'][:-4] + '.png'
        time.sleep(5)

        response = generate_solution(sample)
        response = response[8:].strip()[:-4]
        
        try:
            response_dict = json.loads(response)

            response_dict['id'] = img
            ans_file.write(json.dumps(response_dict) + '\n')
            ans_file.flush()
            
        except json.JSONDecodeError as e:
            # Catch the error and print details to help debug
            print(f"JSONDecodeError: {e}")
            print(f"Invalid response content: {response}")
