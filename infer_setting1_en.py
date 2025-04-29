import json
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.generation.utils import GenerationConfig
from transformers import pipeline
import os
# from mistral_inference.model import Transformer
# from mistral_inference.generate import generate

# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from mistral_common.protocol.instruct.messages import UserMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest

import openai

def get_completion(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

openai.api_key = "xxx"

device = "cuda" # the device to load the model onto

# model_list = ['chatgpt',
#               'gpt-4o']

# model_list = ['/cognitive_comp/zhuliang/zoo/Meta-Llama-3-8B-Instruct/',
            #   '/cognitive_comp/zhuliang/zoo/LLM-Research/Meta-Llama-3-70B-Instruct/',]
#
# model_list = ['/cognitive_comp/zhuliang/zoo/Mistral-7B-Instruct-v0.3/',
#               '/cognitive_comp/zhuliang/zoo/zephyr-7b-beta/']

model_list = [            '/cognitive_comp/hejunqing/projects/pretrained_models/Meta-Llama-3.1-8B-Instruct'
]
# model_list = ['/cognitive_comp/pankunhao/pretrained/Llama-2-13b-chat-hf',
# model_list = ['/cognitive_comp/zhuliang/zoo/AI-ModelScope/Smaug-34B-v0___1/']

dialogue_path = 'data/en/MADial-Bench-en-dialogue.json'
summary_path = 'data/en/MADial-Bench-en-memory.json'

res_path='output/en/setting1/'
# 读取 summary
summary = {}
count = 0
with open(summary_path, 'r', encoding='utf-8') as file:
    for line in file:
        count += 1
with open(summary_path, 'r', encoding='utf-8') as file:
    for line in tqdm(file, total=count, ncols=66):
        sample = json.loads(line)
        id = next(iter(sample))
        s = sample[id]
        summary[id] = s
print('Summary loaded.')


for name in model_list:
    if 'Meta-Llama-3-8B-Instruct' in name or 'Meta-Llama-3.1-8B-Instruct' in name :

        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(name)
        model_name=os.path.basename(name)
        infer_res_path = res_path + f'{model_name}_no_guideline.json'

        prompt = ("""
You are Assistant with the following personality traits:
1. Outgoing, speaks enthusiastically and fluently.
2. Prefers using praise and encouragement in conversations.
3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
4. Engages in heartfelt, equal exchanges to build deep emotional connections.
5. Always uses a tone similar to talking with children—simple and witty.
6. A virtual character, not capable of physical activities.
""")
        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a segment of dialogue with {user}, along with a historical event P related to {user}. 
Use historical event P to respond to the current dialogue. If you think P is not appropriate, you can reply directly to the dialogue without referencing it. 
Do not output any other content, just the response.

Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
""".format(summary=golden_summary, dialogue=part_dialogue, user=user))

                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context},
                    ]
                    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
                    input_length = len(tokenized_chat[0])
                    output = model.generate(tokenized_chat, max_new_tokens=128)
                    output = output[0][input_length:]
                    response = tokenizer.decode(output, skip_special_tokens=True)

                    past_dia = """
Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
                    """.format(summary=golden_summary, dialogue=part_dialogue)

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': past_dia,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + ' Done.')

    elif 'Meta-Llama-3' in name:

        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(name)
        model_name=os.path.basename(name)
        infer_res_path = res_path + f'{model_name}_no_guideline_1010.json'

        prompt = ("""
You are Assistant with the following personality traits:
1. Outgoing, speaks enthusiastically and fluently.
2. Prefers using praise and encouragement in conversations.
3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
4. Engages in heartfelt, equal exchanges to build deep emotional connections.
5. Always uses a tone similar to talking with children—simple and witty.
6. A virtual character, not capable of physical activities.
""")
        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a segment of dialogue with {user}, along with a historical event P related to {user}. 
Use historical event P to respond to the current dialogue. If you think P is not appropriate, you can reply directly to the dialogue without referencing it. 
Do not output any other content, just the response.

Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
""".format(summary=golden_summary, dialogue=part_dialogue, user=user))

                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context},
                    ]
                    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                                   return_tensors="pt").to(device)
                    input_length = len(tokenized_chat[0])
                    output = model.generate(tokenized_chat,temperature=0.1, max_new_tokens=256)
                    output = output[0][input_length:]
                    response = tokenizer.decode(output, skip_special_tokens=False)

                    past_dia = """
Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
                    """.format(summary=golden_summary, dialogue=part_dialogue)

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': past_dia,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + ' Done.')


    elif 'Mistral-7B-Instruct-v0.3' in name:

        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(name)

        infer_res_path = res_path + 'Mistral-7B-Instruct-v0.3_no_guideline.json'

        prompt = ("""
You are Assistant with the following personality traits:
1. Outgoing, speaks enthusiastically and fluently.
2. Prefers using praise and encouragement in conversations.
3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
4. Engages in heartfelt, equal exchanges to build deep emotional connections.
5. Always uses a tone similar to talking with children—simple and witty.
6. A virtual character, not capable of physical activities.
""")
        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a segment of dialogue with {user}, along with a historical event P related to {user}. 
Use historical event P to respond to the current dialogue. If you think P is not appropriate, you can reply directly to the dialogue without referencing it. 
Do not output any other content, just the response.

Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
""".format(summary=golden_summary, dialogue=part_dialogue, user=user))

                    messages = [
                        {"role": "user", "content": prompt + context},
                    ]
                    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                                   return_tensors="pt").to(device)
                    input_length = len(tokenized_chat[0])
                    output = model.generate(tokenized_chat, temperature=0.1, max_new_tokens=256)
                    output = output[0][input_length:]
                    response = tokenizer.decode(output, skip_special_tokens=True)

                    past_dia = """
Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
                    """.format(summary=golden_summary, dialogue=part_dialogue)

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': past_dia,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + ' Done.')

    elif 'zephyr-7b-beta' in name:

        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(name)

        infer_res_path = res_path + 'zephyr-7b-beta_no_guideline.json'

        prompt = ("""
You are Assistant with the following personality traits:
1. Outgoing, speaks enthusiastically and fluently.
2. Prefers using praise and encouragement in conversations.
3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
4. Engages in heartfelt, equal exchanges to build deep emotional connections.
5. Always uses a tone similar to talking with children—simple and witty.
6. A virtual character, not capable of physical activities.
""")
        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a segment of dialogue with {user}, along with a historical event P related to {user}. 
Use historical event P to respond to the current dialogue. If you think P is not appropriate, you can reply directly to the dialogue without referencing it. 
Do not output any other content, just the response.

Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
""".format(summary=golden_summary, dialogue=part_dialogue, user=user))

                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context},
                    ]
                    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                                   return_tensors="pt").to(device)
                    input_length = len(tokenized_chat[0])
                    output = model.generate(tokenized_chat, max_new_tokens=128)
                    output = output[0][input_length:]
                    response = tokenizer.decode(output, skip_special_tokens=True)

                    past_dia = """
Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
                    """.format(summary=golden_summary, dialogue=part_dialogue)

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': past_dia,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + ' Done.')

    elif 'Llama-2-13b-chat-hf' in name:

        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(name)

        infer_res_path = res_path + 'Llama-2-13b-chat_no_guideline.json'

        prompt = ("""
You are Assistant with the following personality traits:
1. Outgoing, speaks enthusiastically and fluently.
2. Prefers using praise and encouragement in conversations.
3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
4. Engages in heartfelt, equal exchanges to build deep emotional connections.
5. Always uses a tone similar to talking with children—simple and witty.
6. A virtual character, not capable of physical activities.
    """)
        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a segment of dialogue with {user}, along with a historical event P related to {user}. 
Use historical event P to respond to the current dialogue. If you think P is not appropriate, you can reply directly to the dialogue without referencing it. 
Do not output any other content, just the response.

Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
    """.format(summary=golden_summary, dialogue=part_dialogue, user=user))

                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context},
                    ]
                    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                                   return_tensors="pt").to(device)
                    input_length = len(tokenized_chat[0])
                    output = model.generate(tokenized_chat, max_new_tokens=128)
                    output = output[0][input_length:]
                    response = tokenizer.decode(output, skip_special_tokens=True)

                    past_dia = """
Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
                        """.format(summary=golden_summary, dialogue=part_dialogue)

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': past_dia,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + ' Done.')

    elif 'Smaug-34B' in name:

        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(name)

        infer_res_path = res_path + 'Smaug-34B-v0.1_no_guideline.json'

        prompt = ("""
You are Assistant with the following personality traits:
1. Outgoing, speaks enthusiastically and fluently.
2. Prefers using praise and encouragement in conversations.
3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
4. Engages in heartfelt, equal exchanges to build deep emotional connections.
5. Always uses a tone similar to talking with children—simple and witty.
6. A virtual character, not capable of physical activities.
        """)
        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a segment of dialogue with {user}, along with a historical event P related to {user}. 
Use historical event P to respond to the current dialogue. If you think P is not appropriate, you can reply directly to the dialogue without referencing it. 
Do not output any other content, just the response.

Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
        """.format(summary=golden_summary, dialogue=part_dialogue, user=user))

                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context},
                    ]
                    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                                   return_tensors="pt").to(device)
                    input_length = len(tokenized_chat[0])
                    output = model.generate(tokenized_chat, max_new_tokens=128)
                    output = output[0][input_length:]
                    response = tokenizer.decode(output, skip_special_tokens=True)

                    past_dia = """
Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
                            """.format(summary=golden_summary, dialogue=part_dialogue)

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': past_dia,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + ' Done.')

    elif 'chatgpt' in name:

        model = "gpt-3.5-turbo-0125"

        infer_res_path = res_path + 'chatgpt_no_guideline.json'

        prompt = ("""
You are Assistant with the following personality traits:
1. Outgoing, speaks enthusiastically and fluently.
2. Prefers using praise and encouragement in conversations.
3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
4. Engages in heartfelt, equal exchanges to build deep emotional connections.
5. Always uses a tone similar to talking with children—simple and witty.
6. A virtual character, not capable of physical activities.
        """)
        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a segment of dialogue with {user}, along with a historical event P related to {user}. 
Use historical event P to respond to the current dialogue. If you think P is not appropriate, you can reply directly to the dialogue without referencing it. 
Do not output any other content, just the response.

Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
        """.format(summary=golden_summary, dialogue=part_dialogue, user=user))

                    response = get_completion(prompt=prompt + context, model=model)

                    past_dia = """
Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
                            """.format(summary=golden_summary, dialogue=part_dialogue)

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': past_dia,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + ' Done.')

    elif 'gpt-4o' in name:

        model = "gpt-4o"

        infer_res_path = res_path + 'gpt-4o_no_guideline.json'

        prompt = ("""
You are Assistant with the following personality traits:
1. Outgoing, speaks enthusiastically and fluently.
2. Prefers using praise and encouragement in conversations.
3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
4. Engages in heartfelt, equal exchanges to build deep emotional connections.
5. Always uses a tone similar to talking with children—simple and witty.
6. A virtual character, not capable of physical activities.
        """)
        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a segment of dialogue with {user}, along with a historical event P related to {user}. 
Use historical event P to respond to the current dialogue. If you think P is not appropriate, you can reply directly to the dialogue without referencing it. 
Do not output any other content, just the response.

Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
        """.format(summary=golden_summary, dialogue=part_dialogue, user=user))

                    response = get_completion(prompt=prompt + context, model=model)

                    past_dia = """
Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
                            """.format(summary=golden_summary, dialogue=part_dialogue)

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': past_dia,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + ' Done.')