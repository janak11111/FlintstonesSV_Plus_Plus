# Import libraries
import transformers
import textwrap
import os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, GenerationConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import json



# Load VSG jsonl file
with open('path/to/flinstone/vsg/file.jsonl') as f:
    test_data = [json.loads(x) for x in f]


# create prompt for Mistral LLM
def generate_story(sample):
    return f"""Create a concise and coherent story of the image using its textual scene graph. 
    The textual scene graph provides the details about objects, their attributes, and the relationships between them. 

    Ensure that story no longer than two lines.

    **Textual Scene Graph**: 

    - **Objects and Attributes**: 
        {sample['objects']}

    - **Relationships**: 
        {sample['relationships']}

    **Generated Story**: """


# set the mistral model
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.3'


# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map='auto')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
print("tokenizer loaded")


# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
)


# set the lora config
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)


# load the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    trust_remote_code=True)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = False)
model = get_peft_model(model, peft_config)
print("model loaded")



# function to generate scene narrative from Mistral LLM
def generate_response(prompt, model):
    encoding = tokenizer(prompt, return_tensors='pt')
    input_ids = encoding["input_ids"]

    generation_config = GenerationConfig(temperature=1, top_p=0.85, repetition_penalty=1.15)

    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids.to('cuda'),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=120,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True
        )

# function to extract the response
def format_response(response):
    decoded_output = tokenizer.decode(response.sequences[0], skip_special_tokens=True)
    story = decoded_output.split('**Generated Story**: ')[1].strip()
    story = story.split('\n\n')
    story = [x.strip() for x in story]
    story = " ".join(story)
    return story


# function to call LLM
def ask_llm(sample, model):
    prompt = generate_story(sample)
    response = generate_response(prompt, model)
    return format_response(response)


new_data = []


# loop to generate Improved Scene Narrative and store it on json file
with open('path/to/output/json/file.jsonl', 'w') as f:
    for i in tqdm(range(len(test_data))):
        prediction = ask_llm(test_data[i], model)

        temp = {}
        temp['id'] = test_data[i]['id']
        temp['story'] = prediction
        temp['objects'] = test_data[i]['objects']
        temp['relationships'] = test_data[i]['relationships']
        

        f.write(json.dumps(temp) + '\n')
        f.flush()   
