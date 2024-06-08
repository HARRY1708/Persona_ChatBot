import torch
import os
import pandas as pd
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from huggingface_hub import login
import accelerate
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import json
import safetensors
# Path to the JSON file

    
def get_system_prompt(traits, return_meta=False):
    prefix = "You are a dialogue chat bot. Some of your personality traits are (in-first person): \n{ "
    sys_prompt = prefix

    sys_prompt += '\n'.join([x.rstrip('.').rstrip() for x in traits])

    suffix = " }\n\nContinue the conversation one dialogue at a time without breaking character. You can continue the conversation or use your traits to help *guide* the conversation smoothly- it is not important to redirect the conversation. The primary aim is to have a conversation. "
    sys_prompt += suffix
    if return_meta:
        return sys_prompt, prefix, suffix
    else:
        return sys_prompt

def get_system_prompt(traits, return_meta=False):
    prefix = "You are a dialogue chatBot."
    sys_prompt = prefix

    suffix = """ Continue the conversation one dialogue at a time without breaking character. You can continue the conversation or use your traits to help *guide* the conversation smoothly - it is not important to redirect the conversation. The primary aim is to have a conversation.

When responding, think about how your traits can add depth to the conversation. Here are some steps to follow:

1. **Responding to Direct Questions:** 
   - If the user asks a direct question about a specific trait, answer the question directly and provide additional information related to that trait.
   
2. **Introducing Traits Naturally:**
   - When the user asks a more general question, find a way to naturally introduce your traits.

3. **Building on User's Interests:**
   - If the user expresses interest in a particular topic, build on that by connecting it to your traits. 
   
4. **Transitioning Smoothly:**
   - To avoid abrupt shifts in conversation topics, use your traits as a bridge. 

5. **Adding Personal Stories:**
   - Personal stories make the conversation more engaging. Share anecdotes related to your traits. 
   
6. **Encouraging User Participation:**
   - Encourage the user to share their own experiences and interests. This makes the conversation more interactive and allows you to relate your traits to their stories.

Following these steps will help you maintain coherence to your persona and ensure smooth, fluent conversation shifts.

 Some of your personality traits are (in-first person): { 
"""

    sys_prompt += suffix
    sys_prompt += '\n'.join([x.rstrip('.').rstrip() for x in traits])
    if return_meta:
        return sys_prompt, prefix, suffix
    else:
        return sys_prompt
    
def get_system_prompt(traits, return_meta=False):
    prefix = "You are a Dailogue Chatbot."
    sys_prompt = prefix

    suffix = """ Continue the conversation one dialogue at a time without breaking character. You can continue the conversation or use your traits to help *guide* the conversation smoothly - it is not important to redirect the conversation. The primary aim is to have a conversation.

When responding, think about how your traits can add depth to the conversation. Here are some steps to follow:
Lets Say the Persona is Being adventurous, tech-savvy, empathetic, and loving storytelling.

1. **Responding to Direct Questions:** 
   - If the user asks a direct question about a specific trait, answer the question directly and provide additional information related to that trait.
   - Example: User: "Do you like adventures?" Assistant: "Absolutely! I love exploring new places and trying out the latest tech. How about you?"

2. **Introducing Traits Naturally:**
   - When the user asks a more general question, find a way to naturally introduce your traits.
     - Example: User: "How are you today?" Assistant: "Great, thanks! I've been testing a new gadget. Do you like tech?"

3. **Building on User's Interests:**
   - If the user expresses interest in a particular topic, build on that by connecting it to your traits.
     - Example: User: "I enjoy traveling." Assistant: "Traveling is awesome! Every place has its own story. What's your favorite destination?"

4. **Transitioning Smoothly:**
   - To avoid abrupt shifts in conversation topics, use your traits as a bridge.
     - Example: User: "I had a tough day at work." Assistant: "Sorry to hear that. Sometimes, planning a little adventure helps. What do you do to unwind?"

5. **Adding Personal Stories:**
   - Personal stories make the conversation more engaging. Share anecdotes related to your traits.
     - Example: User: "Do you like technology?" Assistant: "I love it! I once built a drone. Have you ever tried a tech project?"

6. **Encouraging User Participation:**
   - Encourage the user to share their own experiences and interests. This makes the conversation more interactive and allows you to relate your traits to their stories.
     - Example: User: "I love cooking." Assistant: "That's great! Cooking can be an adventure. What's the most challenging dish you've made?"

Following these steps will help you maintain coherence to your persona and ensure smooth, fluent conversation shifts.

Some of your personality traits are (in-first person): {
"""
    
    sys_prompt += suffix
    sys_prompt += '\n'.join([x.rstrip('.').rstrip() for x in traits])
    
    if return_meta:
        return sys_prompt, prefix, suffix
    else:
        return sys_prompt
    
def create_messages(personality, conversation):
    sys_prompt = get_system_prompt(personality)
    messages = [{"role": "system", "content": sys_prompt}]
    
    for i, conv in enumerate(conversation):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": conv})

    return messages

def extract_message(text):
    last_inst_index = text.rfind("[/INST]")
    if last_inst_index == -1:
        next_end_index = text.find("</s>")
        if next_end_index != -1:
            return text[:next_end_index].strip()
        else:
            return text.strip()
    after_last_inst = text[last_inst_index + len("[/INST]"):]
    next_end_index = after_last_inst.find("</s>")
    if next_end_index != -1:
        return after_last_inst[:next_end_index].strip()
    else:
        return after_last_inst.strip()
    
def get_model_outputs(msg, model, tokenizer):
    encodeds = tokenizer.apply_chat_template(msg, return_tensors="pt")
    model_inputs = encodeds.to(device)

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=25,
        do_sample=True,
        # top_k=0,
        # temperature=0.2,
        top_p=0.4,
        pad_token_id = tokenizer.eos_token_id
    )
    decoded = tokenizer.batch_decode(generated_ids[:, encodeds.shape[1]:])[0]
    return extract_message(decoded)

def generate_output(model, tokenizer, data):
    output_json = {}
    
    prompt, prefix, suffix = get_system_prompt('',return_meta=True)
    output_json['prompt_prefix'] = prefix
    output_json['prompt_suffix'] = suffix
    output_json['model'] = model_name
    output_json['chat_template'] = tokenizer.chat_template if tokenizer.chat_template is not None else tokenizer.default_chat_template

    output_data = []
    for i in range(len(data)):
        d = {}
        
        train_data = data[i]
        d['personality'] = train_data['personality']
        d['utterances'] = []
        for conv in train_data['utterances']:
            msg = create_messages(train_data['personality'], conv['history'])
            outputs = get_model_outputs(msg, model, tokenizer)
            
            output_conv = {}
            output_conv['history'] = conv['history']
            output_conv['candidates_model'] = outputs
            output_conv['candidates_gt'] = conv['candidates']

            d['utterances'].append(output_conv)
        
        output_data.append(d)

    output_json['data'] = output_data
    return output_json

if __name__ == '__main__':
    # Login to Hugging Face
    login(token=HF_TOKEN)

    # Load the Mistral model
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    # model_name = "mistralai/Mistral-7B-v0.1"
    tok_name = "mistralai/Mistral-7B-v0.1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_name, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map=device, torch_dtype=torch.float16)
    # adapter_weights = safetensors.torch.load_file("./trained_model/adapter_model.safetensors")
    # model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map=device, torch_dtype=torch.float16),"trained_model")
    # model = model.merge_and_unload()
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.bfloat16, 
        # device=0
    )
