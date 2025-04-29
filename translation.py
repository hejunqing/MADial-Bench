import os
import json
from tqdm import tqdm
import openai
import ast

openai.api_key = "xxx"
model = "gpt-4-turbo-2024-04-09"
client = openai.OpenAI(api_key='xxx')

en_dialoug_path='data/en/MADial-bench-en-dialogue.json'
def get_completion(prompt, model=model):
    messages = [{"role":"system","content":"你是一名出色的翻译家。"},{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

data_path = 'data/ch/MADial-Bench-ch-dialogue.json'
count = 0
with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        count += 1
with open(data_path, 'r', encoding='utf-8') as file:
    for line in tqdm(file, total=count, ncols=66):
        sample = json.loads(line)
        dialogue = sample['dialogue']
        en_dia = []
        for x in dialogue:
            prompt = ("""
请将下面的内容中的中文翻译为英语，要求如下：
- 符合native speaker的感觉；
- 符号全部换成英文符号，其它特殊符号保持不变；
- 如果没有中文，则直接原样输出，不要输出任何无关内容:
{}""".format(x))
            en_s = get_completion(prompt=prompt, model=model)
            en_dia.append(en_s + '\n')
        sample['dialogue'] = en_dia
        with open('', 'a+', encoding='utf-8') as file:
            json.dump(sample, file, ensure_ascii=False)
            file.write('\n')







































