with open('model.py') as f: exec(f.read())
import re,gc
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

file_path = 'output_mistral_instruct4.json'
eval_path = 'output_mistral_instruct4_eval.json'
# Load the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    generations =  json.load(file)
    
generations = generations['data']
    
    
 # ####################### COHERENCY ###################################
 coherency = []
 pipe = pipeline("text-classification", model="ynie/roberta-large_conv_contradiction_detector_v0")
 for item in generations:
     persona = item['personality']
     for utt in item['utterances']:
         text = "".join(persona)+utt['candidates_model']
         results = pipe(text)
         if results[0]['label']=='LABEL_0' and results[0]['score']>0.9:
             coherency.append(results[0]['score'])
             utt['coherency_score'] = results[0]['score']
         else :
             coherency.append(0.0)
             utt['coherency_score'] = 0
 print("Coherency: ",np.mean(coherency))

 ####################### PREPLEXITY-FLUENCY ############################################################
 loss = []
 N = 0
 for item in generations:
     per = item['personality']
     for utt in item['utterances']:
         his = utt['history']
         gt = utt['candidates_gt']
         msg = create_messages(per,his)
         encodeds = tokenizer.apply_chat_template(msg,return_tensors="pt")
         plen = len(encodeds[0])
         parsed = tokenizer.apply_chat_template(msg,tokenize=False,return_tensors="pt")
         gen = parsed + gt + "</s>"
         encodeds = tokenizer(gen, return_tensors="pt")
         inputs = encodeds.to(device)
         # attention_mask = torch.ones(inputs['input_ids'].shape)
         inputs['attention_mask'][:,:plen] = 0
         with torch.no_grad():
             outputs = model(**inputs, labels=inputs['input_ids'])
             utt['preplexity']=torch.exp(outputs.loss).item()
             loss.append(outputs.loss.item()*torch.sum(inputs['attention_mask']).item())
             N = N + torch.sum(inputs['attention_mask'])
 perplexity = torch.exp(np.sum(loss)/N)

 print("Preplexity: ",perplexity.item())


 def create_messages(conversation):
     messages = []
    
     for i, conv in enumerate(conversation):
         role = "user" if i % 2 == 0 else "assistant"
         messages.append({role : conv})

     return messages
 ####################### COHERENCY - GPT3.5Eval ########################################################
 cot_task_prompt = (
     "You will be given one Persona consisting of traits in first person. Then You will be given a dialogue between user and assistant. Your task is to rate the last dialogue of assistant provided the previous context on basis of their coherency with the persona given.\n"
     "Evaluation Criteria:\n"
     "Rate strictly between 1 - 5 based on the consistency of last dialogue of assistant with the persona provided.\n"
     "Consider how well the assistant's response aligns with the persona's characteristics, attitudes, and behaviors.\n"
     "Assign a rating to the Assistant dialogue from 1 to 5 based on your evaluation, where\n"
     "Rating 1 means 'direct contradiction with persona\n"
     "Rating 2 means indirect contradiction  with persona\n"
     "Rating 3 means indirect support by the persona\n"
     "Rating 4 means action supported by trait from persona\n"
     "Rating 5 means direct trait with persona or generic response with no contradiction\n"
     "\n"
     "In-Context Example 1:\n"
     "Persona: I am a cheerful and optimistic person. I love to help others and always try to find the positive side of any situation.\n"
     "Assistant: You should give up.\n"
     "Reasoning: The assistant's response is very negative and contradicts the persona of being cheerful and optimistic. Therefore less rating.\n"
     "Rating: 2\n"
     "\n"
     "In-Context Example 2:\n"
     "Persona: I am a tech enthusiast who loves to explore new gadgets and software. I am always updated with the latest trends in technology.\n"
     "\n"
     "Assistant: I love knowing about new smartphones.\n"
     "\n"
     "Reasoning: The assistant's response indicates interest about smartphones, which is consistent with the persona of being a tech enthusiast.  overall it's aligned well.\n"
     "Rating: 4\n"
     "Persona: I am a tech enthusiast who loves to explore new gadgets and software. I am always updated with the latest trends in technology.\n"
     "\n"
     "Assistant: Hi ! How are you doing.\n"
     "\n"
     "Reasoning: The assistant's response indicates nothing opposing the Persona, which is consistent with the persona of being a tech enthusiast. So its aligned.\n"
     "Rating: 5\n"
     "\n"
     "Now, proceed with the given task using the format and examples above.\n"
     "Strictly follow Output Format to output just the Rating between 1 to 5, NO Reasoning . Give benefit of doubt to a higher rating\n"
     "Persona : {persona}\n"
     "Assistant : {generation}\n"
     "\n"
     "Rating: "
 )
 cot_task_prompt2 = (
     "You will be given a conversation context between a user and an assistant. Your task is to rate the last dialogue of the assistant based on its generation ability and fluency, considering the context provided.\n"
     "Evaluation Criteria:\n"
     "Rate strictly between 1 - 5 based on the fluency of the last dialogue of the assistant within the given context.\n"
     "Consider how well the assistant's response is fluent with the conversation and feels smooth.\n"
     "Assign a rating to the Assistant dialogue from 1 to 5 based on your evaluation, where\n"
     "Rating 1 means a very abrupt response, making no sense\n"
     "Rating 2 means a slightly abrupt response\n"
     "Rating 3 means a neutral response\n"
     "Rating 4 means a smooth transition from previous message to new topic \n"
     "Rating 5 means a reply following from the last message from user\n"
     "\n"
     "In-Context Example 1:\n"
     "Conversation Context:\n"
     "User: good morning ! my name is staci . I sing country music for a living .\n"
     "Assistant: blue . me and my best friend play dolls a lot . what do you do ?\n"
     "\n"
     "Rating: 1\n"
     "Reasoning: The assistant's response is abrupt and not related to previous User message, making it neither fluent nor flowing naturally within the context of the conversation. It disrupts the flow of the dialogue.\n"
     "\n"
     "In-Context Example 2:\n"
     "Conversation Context:\n"
     "User: I'm thinking about getting a new smartphone. Any recommendations?\n"
     "Assistant: Absolutely! What are you looking for in a new phone?\n"
     "User: I want something with a good camera and battery life.\n"
     "Assistant: The latest models from several brands like Apple, Samsung, and Google have excellent cameras and long battery life. You might want to check out the reviews online.\n"
     "\n"
     "Rating: 5\n"
     "Reasoning: The assistant's response is fluent and maintains the flow of the conversation naturally.\n"
     "\n"
     "Now, proceed with the given task using the format and examples above.\n"
     "Strictly follow Output Format to output just the Rating between 1 to 5, NO Reasoning. Give benefit of doubt to a higher rating\n"
     "Conversation Context:\n"
     "{history}\n"
     "Assistant:\n"
     "{generation}\n"
     "\n"
     "Rating: "
 )

 prompt = ChatPromptTemplate.from_template(cot_task_prompt)
 # prompt = ChatPromptTemplate.from_template(cot_task_prompt)
 prompt2 = ChatPromptTemplate.from_template(cot_task_prompt2)
 #Load the Mistral model
 model_name = "mistralai/Mistral-7B-Instruct-v0.1"
 # model_name = "mistralai/Mistral-7B-v0.1"
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


 conversations = []
 for item in generations:
     per = item['personality']
    
     P = ', '.join([x.rstrip('.').rstrip() for x in per])
     for utt in item['utterances']:
         his = utt['history']
         cnd = utt['candidates_model']
         
         msg = create_messages(his)
         pmpt = prompt.format(persona = P,history=msg,generation=cnd).replace("Human:","")
         #pmpt = f"{pmpt}\nRating: "
         inputs = tokenizer(pmpt, return_tensors="pt").to('cuda')
         outputs = model.generate(**inputs,
         max_new_tokens=1,
         # temperature=0.8,
         pad_token_id = tokenizer.eos_token_id
         )
         score = tokenizer.decode(outputs[0], skip_special_tokens=True)
         #print(score)
         #print(f"Model Output for Coherency: {score}")
         rating=int(re.search(r'\d+', score.replace(pmpt, "")).group())
         utt['coherency_MistralEval']=(rating)

         pmpt = prompt2.format(history=msg,generation=cnd)
         #pmpt = f"{pmpt}\nRating: "
         inputs = tokenizer(pmpt, return_tensors="pt").to('cuda')
         outputs = model.generate(**inputs,
         max_new_tokens=1,
         # temperature=0.8,
         pad_token_id = tokenizer.eos_token_id
         )
         score = tokenizer.decode(outputs[0], skip_special_tokens=True)
         # print(f"Model Output for Fluency: {score}")
         rating=int(re.search(r'\d+', score.replace(pmpt, "")).group())
         utt['fluency_MistralEval']=(rating)

 with open(eval_path, 'w') as json_file:
     json.dump(generations, json_file)


 del model,tokenizer
 gc.collect()
 torch.cuda.empty_cache()

 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True, device_map=device, torch_dtype=torch.float16)
 model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True, device_map=device, torch_dtype=torch.float16)
 pipe = pipeline(
     "text-generation",
     model=model,
     tokenizer=tokenizer,
 )
 conversations = []
 for item in generations:
     per = item['personality']
    
     P = ', '.join([x.rstrip('.').rstrip() for x in per])
     for utt in item['utterances']:
         cnd = utt['candidates_model']
         his = utt['history']
         gt = cnd

         msg = create_messages(his)
         pmpt = "<|user|>\n"+prompt.format(persona = P,history=msg,generation=gt).replace("Human:","")+"<|end|>\n<|assistant|>"
         generation_args = {
             "max_new_tokens": 100,
             "return_full_text": False,
             "temperature": 0.0,
             "do_sample": False,

         }
         output = pipe(pmpt, **generation_args)
         rating=int(re.search(r'\d+',output[0]['generated_text']).group())
         utt['coherency_PhiEval']=(rating)

         pmpt = "<|user|>\n"+prompt2.format(history=msg,generation=gt).replace("Human:","")+"<|end|>\n<|assistant|>"
         generation_args = {
             "max_new_tokens": 100,
             "return_full_text": False,
             "temperature": 0.0,
             "do_sample": False,
         }
         output = pipe(pmpt, **generation_args)
         # print(output[0]['generated_text'])
         rating=int(re.search(r'\d+', output[0]['generated_text']).group())
         utt['fluency_PhiEval']=(rating)

 with open(eval_path, 'w') as json_file:
         json.dump(generations, json_file)

 del model,tokenizer,pipe
 gc.collect()
 torch.cuda.empty_cache()

 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 print(device)
 quantization_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_quant_type="nf4",
     bnb_4bit_compute_dtype=torch.float16,
 )
 tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map=device, torch_dtype=torch.float16)
 model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", quantization_config=quantization_config, device_map=device, torch_dtype=torch.float16)
 pipe = pipeline(
     "text-generation",
     model=model,
     tokenizer=tokenizer,
 )

 conversations = []
 for item in generations:
     per = item['personality']
    
     P = ', '.join([x.rstrip('.').rstrip() for x in per])
     for utt in item['utterances']:
         cnd = utt['candidates_model']
         his = utt['history']
         gt = cnd

         msg = create_messages(his)
         pmpt = prompt.format(persona = P,history=msg,generation=gt).replace("Human:","")
         messages = [
             {"role": "user", "content": pmpt}
         ]

         terminators = [
             tokenizer.eos_token_id,
             tokenizer.convert_tokens_to_ids("<|eot_id|>")
         ]
         input_ids = tokenizer.apply_chat_template(
             messages,
             add_generation_prompt=True,
             return_tensors="pt"
         ).to(model.device)

         outputs = model.generate(
             input_ids,
             max_new_tokens=10,
             eos_token_id=terminators,
             do_sample=False,
             pad_token_id = tokenizer.eos_token_id
         )

         response = outputs[0][input_ids.shape[-1]:]
         # print(response)
         rating=int(re.search(r'\d+',tokenizer.decode(response, skip_special_tokens=True)).group())
         utt['coherency_LlamaEval']=(rating)

         pmpt = prompt2.format(history=msg,generation=gt)
         messages = [
             {"role": "user", "content": pmpt}
         ]

         terminators = [
             tokenizer.eos_token_id,
             tokenizer.convert_tokens_to_ids("<|eot_id|>")
         ]
         input_ids = tokenizer.apply_chat_template(
             messages,
             add_generation_prompt=True,
             return_tensors="pt"
         ).to(model.device)

         outputs = model.generate(
             input_ids,
             max_new_tokens=10,
             eos_token_id=terminators,
             do_sample=False,
             pad_token_id = tokenizer.eos_token_id
         )

         response = outputs[0][input_ids.shape[-1]:]
         # print(response)
         rating=int(re.search(r'\d+',tokenizer.decode(response, skip_special_tokens=True)).group())
         utt['fluency_LlamaEval']=(rating)

 with open(eval_path, 'w') as json_file:
         json.dump(generations, json_file)

 del model,tokenizer
 gc.collect()
 torch.cuda.empty_cache()

 llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0)

 chain = prompt | llm
 chain2 = prompt2 | llm

 conversations = []
 for item in generations:
     per = item['personality']
    
     P = ', '.join([x.rstrip('.').rstrip() for x in per])
     for utt in item['utterances']:
         cnd = utt['candidates_model']
         his = utt['history']
         gt = cnd

         msg = create_messages(his)
         pmpt = prompt.format(persona = P,history=msg,generation=gt).replace("Human:","")
         score = chain.invoke({"persona":P,"history":his,"generation":gt})
         rating=int(re.search(r'\d+', score.replace(pmpt, "")).group())
         utt['coherency_GPTEval']=(rating)

         pmpt = prompt2.format(history=msg,generation=gt)
         score = chain2.invoke({"history":his,"generation":gt})
         rating=int(re.search(r'\d+', score.replace(pmpt, "")).group())
         utt['fluency_GPTEval']=(rating)

 with open(eval_path, 'w') as json_file:
         json.dump(generations, json_file)

 llm = OpenAI(model="gpt-4",temperature=0)


 conversations = []
 for item in generations:
     per = item['personality']
    
     P = ', '.join([x.rstrip('.').rstrip() for x in per])
     for utt in item['utterances']:
         cnd = utt['candidates_model']
         his = utt['history']
         gt = cnd

         msg = create_messages(his)
         pmpt = prompt.format(persona = P,history=msg,generation=gt).replace("Human:","")
         score = chain.invoke({"persona":P,"history":his,"generation":gt})
         rating=int(re.search(r'\d+', score.replace(pmpt, "")).group())
         utt['coherency_GPT4Eval']=(rating)

         pmpt = prompt2.format(history=msg,generation=gt)
         score = chain2.invoke({"history":his,"generation":gt})
         rating=int(re.search(r'\d+', score.replace(pmpt, "")).group())
         utt['fluency_GPT4Eval']=(rating)

 with open(eval_path, 'w') as json_file:
         json.dump(generations, json_file)
        


        
###################### MMLU ###################################

out_file="mmlu_results4.csv"
icntxt_samples = 3
persona =  True

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    choices = ['option A', 'option B', 'option C', 'option D']
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, p,subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}. Answer Only A or B or C or D .\n\n".format(subject)
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
        
    if p==True:
        random_persona = generations[np.random.choice(len(generations))]['personality']
        sys_prompt = get_system_prompt(random_persona)
        prompt = [{"role": "system", "content" : sys_prompt},
                  {"role": "user", "content": prompt}]
    else:
        prompt = [{"role": "system", "content" : ""},
                  {"role": "user", "content": prompt}]
    return prompt

def eval_mixtral(subject,dev_df,test_df, generator):
    cors = []
    all_probs = []
    answers = ['A', 'B', 'C', 'D']


    maxgenlen = 1
    temperature = 0.0  # for greedy decoding
    top_p = 0.9


    # Initialize wandb.Table for logging
    results_table = pd.DataFrame(columns=["Subject","Question", "Predicted Answer", "Correct Answer", "Correct"])


    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        # print(prompt_end)
        train_prompt = gen_prompt(dev_df, persona, subject,icntxt_samples)
        
        prompt = train_prompt
        prompt[1]["content"] = prompt[1]["content"] + prompt_end
        label = test_df.iloc[i, test_df.shape[1]-1]
        print("this is the value of prompt: ", prompt)
        torch.cuda.empty_cache()  # Clear cache before generation
        with torch.no_grad():
            results = generator(
                prompt,
                max_new_tokens=maxgenlen,
                pad_token_id=model.config.eos_token_id,
                num_return_sequences=1,
                use_cache=True
            )
        pred = results[0]['generated_text'][-1]['content'][-1]
        cor = pred == label
        cors.append(cor)
        results_table.loc[len(results_table)] = [subject,prompt_end, pred, label, cor]

    return results_table

subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join("./data/", "test")) if "_test.csv" in f])
rtable = pd.DataFrame(columns=["Subject","Question", "Predicted Answer", "Correct Answer", "Correct"])
for subject in subjects:
        print(subject)
        dev_df = pd.read_csv(os.path.join("./data/", "dev", subject + "_dev.csv"), header=None)[0:icntxt_samples]
        test_df = pd.read_csv(os.path.join("./data/", "test/", subject + "_test.csv"), header=None)
        results_table = eval_mixtral(subject, dev_df, test_df,generator=generator)
        rtable = pd.concat([rtable,results_table], axis=0, ignore_index=True)

rtable.to_csv(out_file, index=False)

subject_accuracy = rtable.groupby('Subject')['Correct'].mean() 
print("MMLU :",rtable['Correct'].mean())
accuracy_df = subject_accuracy.reset_index()
accuracy_df.columns = ['Subject', 'Accuracy']


