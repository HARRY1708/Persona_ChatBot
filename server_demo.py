import gradio as gr

with open('model.py') as f: exec(f.read())

def get_model_outputs(msg, model, tokenizer):
    encodeds = tokenizer.apply_chat_template(msg, return_tensors="pt")
    model_inputs = encodeds.to(device)

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=0,
        temperature=0.6,
        top_p=0.92,
    )
    decoded = tokenizer.batch_decode(generated_ids[:, encodeds.shape[1]:])[0]
    return decoded

def restructure_history(history):
    restructure_history = []
    for i in range(len(history)-1):
        restructure_history.append(history[i][0])
        restructure_history.append(history[i][1])
    restructure_history.append(history[-1])
    return restructure_history

def generate_response(persona_input, prompt_input, history):
    
    history.append(prompt_input)
    
    formatted_history = restructure_history(history)
    
    messages = create_messages([persona_input], formatted_history)
    
    response = get_model_outputs(messages, model, tokenizer)
    
    history.pop()
    
    history.append((prompt_input, response))
    return response, history

def reset_prompt_input():
    return []

with gr.Blocks() as demo:
    gr.Markdown("## Mistral Instruct Model with Persona - Continuous Chat")

    chatbot = gr.Chatbot(height=240)
    persona_input = gr.Textbox(label="Persona", placeholder="Describe the persona here...")
    prompt_input = gr.Textbox(label="Prompt", placeholder="Enter the prompt here...")
    output = gr.Textbox(label="Response")

    generate_button = gr.Button("Generate Response")
    reset_button = gr.ClearButton(components=[prompt_input, chatbot], value="Reset Conversation")
    
    clear_input_prompt = gr.ClearButton(components=[prompt_input, chatbot], value="Reset Conversation")

    # history_state = gr.State([])
    
    # reset_input_prompt = gr.ClearButton.click(generate_response, prompt_input, prompt_input)
    generate_button.click(generate_response, [persona_input, prompt_input, chatbot], [output, chatbot])
    prompt_input.submit(generate_response, [persona_input, prompt_input, chatbot], [output, chatbot])
    # reset_button.click(reset_history, None, history_state)

demo.launch(share=True)
