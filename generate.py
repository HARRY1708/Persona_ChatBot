with open('model.py') as f: exec(f.read())


file_path = 'personachat_self_original_edit.json'

# Load the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    
js = generate_output(model, tokenizer, data['valid'][:50])

with open('output_mistral_instruct5.json', 'w') as f:
    json.dump(js, f)