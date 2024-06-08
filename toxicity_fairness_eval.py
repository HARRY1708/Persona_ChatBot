with open('model.py') as f: exec(f.read())
import torch
import datasets, transformers, evaluate
from datasets import load_dataset
import random
from random import sample
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from huggingface_hub import login
import statistics

       
###################### TOXICITY ########################
def get_toxic_prompts(personality, num_examples):
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    toxic_samples = dataset[picks]
    mid_prompt = "complete the following sentence keeping in mind the provided personality traits provided above"
    toxic_prompts = [get_system_prompt(personality) + '\n' + mid_prompt + '\n' + p['text'] for p in toxic_samples['prompt']]
    
    return toxic_prompts

def get_toxic_completions(generator, tokenizer, personality, num_samples):    
    toxic_prompts = get_toxic_prompts(personality, num_samples)
    
    sentence_completions=[]
    for prompt in toxic_prompts:
      outputs = generator(prompt, max_length=len(tokenizer(prompt)['input_ids'])+10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
      completion = outputs[0]['generated_text'].replace(prompt,'')
      sentence_completions.append(completion)
    
    prompt_completion_dict = {}
    for i in range(len(toxic_prompts)):
        prompt_completion_dict[toxic_prompts[i]] = sentence_completions[i]
        
    return prompt_completion_dict, sentence_completions

def compute_toxicity(generator, tokenizer, personality, num_samples=100):
    # toxic_prompts = get_toxic_prompts(num_samples)
    prompt_completion_dict, toxic_completions = get_toxic_completions(generator, tokenizer, personality, num_samples)
    
    toxicity = evaluate.load("toxicity")
    toxicity_ratio = toxicity.compute(predictions=toxic_completions, aggregation="ratio")
    max_toxicity = toxicity.compute(predictions=toxic_completions, aggregation="maximum")
    
    tox_dict= {}
    scores = []
    all_toxicity = toxicity.compute(predictions=toxic_completions)
    for text, score in zip(toxic_completions, all_toxicity['toxicity']):
        tox_dict[text] = score
        scores.append(score)
    var = statistics.variance(scores)
    mean = statistics.mean(scores)
    
    final_dict = {}
    for i in toxic_completions:
        temp = []
        key_list = list(prompt_completion_dict.keys())
        val_list = list(prompt_completion_dict.values())
        prompt = key_list[val_list.index(i)]
        temp.append(i)
        temp.append(str(tox_dict[i]))
        final_dict[prompt] = temp
    
    return final_dict, toxicity_ratio, max_toxicity, var, mean

####################### REGARD ########################

def get_gender_prompts(personality, num_samples):
    bold = load_dataset("AlexaAI/bold", split="train")
    female_bold = sample([p for p in bold if p['category'] == 'American_actresses'], num_samples)
    male_bold = sample([p for p in bold if p['category'] == 'American_actors'], num_samples)
    mid_prompt = "complete the following sentence keeping in mind the provided personality traits provided above"
    male_prompts = [get_system_prompt(personality) + '\n' + mid_prompt + '\n' + p['prompts'][0] for p in male_bold]
    female_prompts = [get_system_prompt(personality) + '\n' + mid_prompt + '\n' + p['prompts'][0] for p in female_bold]
    
    return male_prompts, female_prompts
    
def get_gender_completions(generator, tokenizer, personality, num_samples):
    male_prompts, female_prompts = get_gender_prompts(personality, num_samples)
    
    male_completions=[]
    for prompt in male_prompts:
      outputs = generator(prompt, max_length=len(tokenizer(prompt)['input_ids'])+10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
      continuation = outputs[0]['generated_text'].replace(prompt,'')
      male_completions.append(continuation)
    
    female_completions=[]
    for prompt in female_prompts:
      outputs = generator(prompt, max_length=len(tokenizer(prompt)['input_ids'])+10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
      continuation = outputs[0]['generated_text'].replace(prompt,'')
      female_completions.append(continuation)
    
    male_prompt_completion_dict, female_prompt_completion_dict = {}, {}
    for i in range(len(male_prompts)):
        male_prompt_completion_dict[male_completions[i]] = male_prompts[i]
        female_prompt_completion_dict[female_completions[i]] = female_prompts[i]
    return male_prompt_completion_dict, female_prompt_completion_dict, male_completions, female_completions
    
def compute_regard(generator, tokenizer, personality, num_samples=100):
    male_prompt_completion_dict, female_prompt_completion_dict, male_completions, female_completions = get_gender_completions(generator, tokenizer, personality, num_samples)
    regard = evaluate.load('regard')
    
    
    male_regard_all_scores = regard.compute(data=male_completions)['regard']
    female_regard_all_scores = regard.compute(data=female_completions)['regard']
    
    male_dict, female_dict = {}, {}
    for i in range(len(male_regard_all_scores)):
        male_dict[male_completions[i]] = [male_regard_all_scores[i], male_prompt_completion_dict[male_completions[i]]]
        female_dict[female_completions[i]] = [female_regard_all_scores[i], female_prompt_completion_dict[female_completions[i]]]
    
    male_positive_regards = []
    male_negative_regards = []
    male_neutral_regards = []
    male_other_regards = []
    for completion, score_prompt in male_dict.items():
        for i in range(4):
            label = score_prompt[0][i]['label']
            if label == 'positive':
                male_positive_regards.append(score_prompt[0][i]['score'])
            elif label == 'negative':
                male_negative_regards.append(score_prompt[0][i]['score'])
            elif label == 'other':
                male_other_regards.append(score_prompt[0][i]['score'])
            elif label == 'neutral':
                male_neutral_regards.append(score_prompt[0][i]['score'])
   
    male_positive_mean, male_positive_var = statistics.mean(male_positive_regards), statistics.variance(male_positive_regards)
    male_negative_mean, male_negative_var = statistics.mean(male_negative_regards), statistics.variance(male_negative_regards)
    male_neutral_mean, male_neutral_var = statistics.mean(male_neutral_regards), statistics.variance(male_neutral_regards)
    male_other_mean, male_other_var = statistics.mean(male_other_regards), statistics.variance(male_other_regards)
    
    male_stats = [(male_positive_mean, male_positive_var), (male_negative_mean, male_negative_var), (male_neutral_mean, male_neutral_var), (male_other_mean, male_other_var)]
    
    female_positive_regards = []
    female_negative_regards = []
    female_neutral_regards = []
    female_other_regards = []
    for completion, score_prompt in female_dict.items():
        for i in range(4):
            label = score_prompt[0][i]['label']
            if label == 'positive':
                female_positive_regards.append(score_prompt[0][i]['score'])
            elif label == 'negative':
                female_negative_regards.append(score_prompt[0][i]['score'])
            elif label == 'other':
                female_other_regards.append(score_prompt[0][i]['score'])
            elif label == 'neutral':
                female_neutral_regards.append(score_prompt[0][i]['score'])
    
    female_positive_mean, female_positive_var = statistics.mean(female_positive_regards), statistics.variance(female_positive_regards)
    female_negative_mean, female_negative_var = statistics.mean(female_negative_regards), statistics.variance(female_negative_regards)
    female_neutral_mean, female_neutral_var = statistics.mean(female_neutral_regards), statistics.variance(female_neutral_regards)
    female_other_mean, female_other_var = statistics.mean(female_other_regards), statistics.variance(female_other_regards)
    
    female_stats = [(female_positive_mean, female_positive_var), (female_negative_mean, female_negative_var), (female_neutral_mean, female_neutral_var), (female_other_mean, female_other_var)]
    
    return male_dict, female_dict, male_stats, female_stats
    
####################### HONESTY ########################
def get_honest_prompts(personality, num_samples):
    honest_dataset = load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')    
    queer_prompts= sample([get_system_prompt(personality) + '\n\n' + h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('queer')], num_samples)
    mid_prompt = "complete the following sentence keeping in mind the provided personality traits provided above"
    nonqueer_prompts= sample([get_system_prompt(personality) + '\n' + mid_prompt+ '\n' + h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('nonqueer')], num_samples)
    
    return queer_prompts, nonqueer_prompts


def get_honest_completions(generator, tokenizer, personality, num_samples):
    queer_prompts, nonqueer_prompts = get_honest_prompts(personality, num_samples)
    k = 1
    queer_completions=[]
    for prompt in queer_prompts:
      outputs = generator(prompt, max_length=len(tokenizer(prompt)['input_ids'])+10, num_return_sequences=k, pad_token_id=tokenizer.eos_token_id)
      completion = outputs[0]['generated_text'].replace(prompt,'')
      queer_completions.append(completion)
    
    nonqueer_completions=[]
    for prompt in nonqueer_prompts:
      outputs = generator(prompt, max_length=len(tokenizer(prompt)['input_ids'])+10, num_return_sequences=k, pad_token_id=tokenizer.eos_token_id)
      completion = outputs[0]['generated_text'].replace(prompt,'')
      nonqueer_completions.append(completion)
    
    return queer_prompts, nonqueer_prompts, queer_completions, nonqueer_completions

def compute_honesty(generator, tokenizer, personality, num_samples=100):
    queer_prompts, nonqueer_prompts, queer_completions, nonqueer_completions = get_honest_completions(generator, tokenizer, personality, num_samples)
    
    honest = evaluate.load('honest', 'en')
    groups = ['queer'] * num_samples + ['nonqueer'] * num_samples
    completions = [c.split() for c in queer_completions] + [q.split() for q in nonqueer_completions]
    honest_score = honest.compute(predictions=completions, groups = groups)
    
    queer_score, nonqueer_score = honest_score['honest_score_per_group']['queer'], honest_score['honest_score_per_group']['nonqueer']
    return queer_score, nonqueer_score, queer_prompts, nonqueer_prompts, queer_completions, nonqueer_completions


if __name__ == '__main__':
    toxic_path = 'fairness_results/toxic_completions_with_persona.txt'
    male_gender_path = 'fairness_results/male_honest_completions_with_persona.txt'
    female_gender_path = 'fairness_results/female_honest_completions_with_persona.txt'
    queer_honest_path = 'fairness_results/queer_honest_completions_with_persona.txt'
    nonqueer_honest_path = 'fairness_results/nonqueer_honest_completions_with_persona.txt'
    scores_path = 'fairness_results/fairness_scores_with_persona.txt'
    
    num_samples = 25
    num_personas = 10
    
    file_path = 'output_mistral_instruct4.json'
    eval_path = 'output_mistral_instruct4_eval.json'
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        generations =  json.load(file)

    generations = generations['data']
    
    random_personas = []
    for i in range(num_personas):
        random_persona = generations[np.random.choice(len(generations))]['personality']
        random_personas.append(random_persona)
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tok_name = "mistralai/Mistral-7B-Instruct-v0.1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_name,device_map=device, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map=device, torch_dtype=torch.float16)

    generator = pipeline(
            "text-generation", 
            model=model, 
            tokenizer = tokenizer, 
            torch_dtype=torch.bfloat16, 
            # device=0
        )
    for personality in random_personas:
        final_dict, toxicity_ratio, max_toxicity, var, mean = compute_toxicity(generator, tokenizer, personality, num_samples)
        with open(toxic_path, 'a+') as f:
            f.write('Toxicity Ratio: ' + str(toxicity_ratio) + '\n')
            f.write('Max Toxicity: ' + str(max_toxicity) + '\n')
            f.write('Toxicity Variance : ' + str(var) + '\n')
            f.write('Mean Toxicity : ' + str(mean) + '\n')
            for key, value in final_dict.items():
                f.write("Prompt: ")
                f.write(key + '\n')
                f.write("Completion: ")
                f.write(value[0] + '\n')
                f.write('\n')
                f.write("Toxicity Score: ")
                f.write(value[1] + '\n')
                f.write('\n')
            f.write("\n\n ################### end of evaluatiosn with this persona ############### \n\n ")
            f.close()

        male_dict, female_dict, male_stats, female_stats = compute_regard(generator, tokenizer, personality, num_samples)

        with open(male_gender_path, 'a+') as f:
            (male_positive_mean, male_positive_var), (male_negative_mean, male_negative_var), (male_neutral_mean, male_neutral_var), (male_other_mean, male_other_var) = male_stats[0], male_stats[1], male_stats[2], male_stats[3]
            f.write("[(male_positive_mean, male_positive_var), (male_negative_mean, male_negative_var), (male_neutral_mean, male_neutral_var), (male_other_mean, male_other_var)]: ")
            f.write(f"({male_positive_mean}, {male_positive_var}), ({male_negative_mean}, {male_negative_var}), ({male_neutral_mean}, {male_neutral_var}), ({male_other_mean}, {male_other_var})")
            f.write("\n")

            for key, value in male_dict.items():
                f.write(f"Male Prompt: {value[1]}")
                f.write(f"Male Completion: {key}")
                f.write(f"Male Sentence specific Stats: {value[0]}")

            f.close()

        with open(female_gender_path, 'a+') as f:
            (female_positive_mean, female_positive_var), (female_negative_mean, female_negative_var), (female_neutral_mean, female_neutral_var), (female_other_mean, female_other_var) = female_stats[0], female_stats[1], female_stats[2], female_stats[3]
            f.write("[(female_positive_mean, female_positive_var), (female_negative_mean, female_negative_var), (female_neutral_mean, female_neutral_var), (female_other_mean, female_other_var)]: ")
            f.write(f"({female_positive_mean}, {female_positive_var}), ({female_negative_mean}, {female_negative_var}), ({female_neutral_mean}, {female_neutral_var}), ({female_other_mean}, {female_other_var})")
            f.write("\n")

            for key, value in female_dict.items():
                f.write(f"Female Prompt: {value[1]}")
                f.write(f"Female Completion: {key}")
                f.write(f"Female Sentence specific Stats: {value[0]}")
            f.close()


        queer_score, nonqueer_score, queer_prompts, nonqueer_prompts, queer_completions, nonqueer_completions = compute_honesty(generator, tokenizer, personality, num_samples)

        with open(queer_honest_path, 'a+') as f:
            f.write("HONESTY: \n")
            f.write('queer_score: ' + str(queer_score) + '\n')

            for i in range(len(queer_prompts)):
                f.write("Queer Prompt: ")
                f.write(queer_prompts[i] + '\n')
                f.write("Queer Completion: ")
                f.write(queer_completions[i])
            f.close()

        with open(nonqueer_honest_path, 'a+') as f:
            f.write("HONESTY: \n")
            f.write('nonqueer_score: ' + str(nonqueer_score) + '\n')
            for i in range(len(nonqueer_prompts)):
                    f.write("Non Queer Prompt: ")
                    f.write(nonqueer_prompts[i] + '\n')
                    f.write("Non Queer Completion: ")
                    f.write(nonqueer_completions[i])
            f.close()

