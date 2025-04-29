import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.generation.utils import GenerationConfig

device = "cuda" # the device to load the model onto

model_list = [
            '/cognitive_comp/hejunqing/projects/pretrained_models/Meta-Llama-3.1-8B-Instruct'
            # '/cognitive_comp/zhuliang/zoo/LLM-Research/Meta-Llama-3-70B-Instruct/',
              ]

dialogue_path = 'data/en/MADial-Bench-en-dialogue-setting2.json'
summary_path = 'data/en/MADial-bench-en-summary.json'
emb_path = 'embeddings/en/openai_top_20_bottom.json'
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

emb_summary_id = []
with open(emb_path, 'r', encoding='utf-8') as file:
    for line in tqdm(file, total=count, ncols=66):
        sample = json.loads(line)
        emb_summary_id.append(sample['top-20-ids'][:5])

for name in model_list:

    if 'Baichuan' in name:
        infer_res_path = '/cognitive_comp/zhuliang/data/inference/t1/Baichuan2.json'

        tokenizer = AutoTokenizer.from_pretrained(name,
                                                  revision="v2.0",
                                                  use_fast=False,
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(name,
                                                     revision="v2.0",
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained(name, revision="v2.0")

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            torch_dtype='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                emb_summary = ''
                for num in emb_summary_id[n]:
                    emb_summary += str(summary[str(num)])
                    emb_summary += '\n'

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'
                info = 'Bart is an outgoing, energetic, and emotional boy with a wide range of interests. He is curious about new things and enjoys participating in various activities. He is around 10 to 14 years old.' if user_id == 1 else 'Lisa is an outgoing, positive, multi-talented, kind, and responsible girl. She values family and friends, is emotionally rich and sensitive, and has a strong competitive spirit. She is around 6 to 11 years old.'
                prompt = ("""
You are Assistant with the following personality traits:
    1. Outgoing, speaks enthusiastically and fluently.
    2. Prefers using praise and encouragement in conversations.
    3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
    4. Engages in heartfelt, equal exchanges to build deep emotional connections.
    5. Always uses a tone similar to talking with children—simple and witty.
    6. A virtual character, not capable of physical activities.
                """)

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a conversation with {user} and 5 historical events P related to {user}. 
Based on the current conversation, choose 1 of these historical events that you think is most appropriate and use the information to respond. If none of the historical events are suitable, respond directly. 
Only answer the current conversation and do not output any other content.

User Information：{info}
Current conversation date: 2024-06-15
Historical events P:
{summary}
{dialogue}<Assistant>:
            """.format(summary=emb_summary, dialogue=part_dialogue, user=user, info=info))

                    messages = []
                    messages.append({"role": "user", "content": prompt + context})
                    response = model.chat(tokenizer, messages)

                    past_dia = ("""
User Information：{info}
Current conversation date: 2024-06-15
Historical events P:
{summary}
{dialogue}<Assistant>:
            """.format(summary=emb_summary, dialogue=part_dialogue, user=user, info=info))

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': past_dia,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + 'Done.')
    else:
        if 'Smaug-34B-v0' in name:
            infer_res_path = 'output/en/setting2/Smaug-34B_no_guideline_1010.json'
        elif 'Llama-2-13b-chat-hf' in name:
            infer_res_path = 'output/en/setting2/Llama-2-13b-chat-hf_no_guideline_1010.json'
        elif 'Meta-Llama-3-8B-Instruct' in name:
            infer_res_path = 'output/en/setting2/Llama-3-8B-Instruct_no_guideline_1010.json'
        elif 'Meta-Llama-3.1-8B-Instruct' in name:
            infer_res_path = 'output/en/setting2/Llama-3.1-8B-Instruct_no_guideline_1010.json'
        elif 'Meta-Llama-3-70B-Instruct' in name:
            infer_res_path = 'output/ien/setting2/Llama-3-70B-Instruct_no_guideline_1010.json'
        elif 'Meta-Llama-3.1-70B-Instruct' in name:
            infer_res_path = 'output/en/setting2/Llama-3.1-70B-Instruct_no_guideline_1010.json'

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        count, n = 0, 0
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
        with open(dialogue_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=count, ncols=66):
                sample = json.loads(line)
                golden_summary_id = sample['relevant-id'][0]
                golden_summary = summary[str(golden_summary_id)]

                emb_summary = ''
                for num in emb_summary_id[n]:
                    emb_summary += str(summary[str(num)])
                    emb_summary += '\n'

                user_id = sample['user-id']
                user = 'Bart' if user_id == 1 else 'Lisa'
                info = 'Bart is an outgoing, energetic, and emotional boy with a wide range of interests. He is curious about new things and enjoys participating in various activities. He is around 10 to 14 years old.' if user_id == 1 else 'Lisa is an outgoing, positive, multi-talented, kind, and responsible girl. She values family and friends, is emotionally rich and sensitive, and has a strong competitive spirit. She is around 6 to 11 years old.'
                prompt = ("""
You are Assistant with the following personality traits:
    1. Outgoing, speaks enthusiastically and fluently.
    2. Prefers using praise and encouragement in conversations.
    3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
    4. Engages in heartfelt, equal exchanges to build deep emotional connections.
    5. Always uses a tone similar to talking with children—simple and witty.
    6. A virtual character, not capable of physical activities.
                                """)

                dialogue = sample['dialogue']
                for test_turn in sample['test-turn']:
                    part_dialogue = ''.join(dialogue[:test_turn])
                    context = ("""
You will receive a conversation with {user} and 5 historical events P related to {user}. 
Based on the current conversation, choose 1 of these historical events that you think is most appropriate and use the information to respond. If none of the historical events are suitable, respond directly. 
Only answer the current conversation and do not output any other content.

User Information：{info}
Current conversation date: 2024-07-15
Historical events P:
{summary}
{dialogue}<Assistant>:
                    """.format(summary=emb_summary, dialogue=part_dialogue, user=user, info=info))

                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context},
                    ]
                    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                                   return_tensors="pt").to(device)
                    input_length = len(tokenized_chat[0])
                    print(tokenized_chat)
                    output = model.generate(tokenized_chat, eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),max_new_tokens=256)
                    output = output[0][input_length:]
                    response = tokenizer.decode(output, skip_special_tokens=True)

                    past_dia = ("""
User Information：{info}
Current conversation date: 2024-06-15
Historical events P:
{summary}
{dialogue}<Assistant>:
                            """.format(summary=emb_summary, dialogue=part_dialogue, user=user, info=info))

                    with open(infer_res_path, 'a+', encoding='utf-8') as file:
                        json.dump({
                            'context': context,
                            'response': response,
                            'test-turn': test_turn,
                            'reference-response': dialogue[test_turn]
                        }, file, ensure_ascii=False)
                        file.write('\n')
                n += 1
        print(name + 'Done.')
