#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sidebyside.py
@Time    :   2024/08/09 16:37:27
@Author  :   He Junqing 
@Version :   1.0
@Contact :   hejunqing@idea.edu.cn
@Desc    :   None
'''

import json
from tqdm import tqdm
import pandas as pd
import os
from process_ano import read_json


summary_path = 'data/ch/MADial-Bench-zh-memory.json'
summarys = {}
count, n = 0, 0
with open(summary_path, 'r', encoding='utf-8') as file:
    for line in file:
        count += 1
with open(summary_path, 'r', encoding='utf-8') as file:
    for line in tqdm(file, total=count, ncols=66):
        sample = json.loads(line)
        id = next(iter(sample))
        s = sample[id]
        summarys[id] = s
print('Summary loaded.')


dialog_path='data/ch/MADial-Bench-zh-dialogue.json'

infer_res_path=[
                'output/ch/Qwen-72B-Instruct_without_guide_0711.json',
                'output/ch/Qwen-72B-Instruct_without_guide_mem_0808.json',]
                # '/output/ch/t3-gpt4-turbo-without_guide_0808.json',
                # 'output/ch/t3-gpt4-turbo-without_guide_mem_0808.json',
                # 'output/ch/t3_glm4_api_0808_without_guide.json',
                # 'output/ch/glm4_api_0809_without_guide_mem.json',
                # 'output/ch/actor_t3_result_0808_without_guide.json',
                # 'output/ch/actor_result_0809_without_guide_mem.json',
                # 'output/ch/doubao_t3_result_0808_without_guide.json',
                # 'output/ch/doubao_t3_result_0809_without_guide_mem.json']

infer_res = [[] for _ in infer_res_path]
for i in range(len(infer_res_path)):
    with open(infer_res_path[i], 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=count, ncols=66):
            sample = json.loads(line)
            infer_res[i].append(sample['response'])
print(len(infer_res),infer_res[0][0])

user_name = ['<Bart>:', '<Lisa>:']
count = 0
with open(dialog_path, 'r', encoding='utf-8') as file:
    for line in file:
        count += 1
# with open(dialog_path, 'r', encoding='utf-8') as file:
for i in range(len(infer_res_path)//2):
    print(i)
    k, n = 0, 0
    model_name=os.path.basename(infer_res_path[i*2]).split('_')[0]
    anno_out_file=f'output/sbs_{model_name}_without_guide_0820.json'
    data=read_json(dialog_path)
    for sample in data:
            test_turn = sample['test-turn']
            user_id = sample['user-id']
            golden_summary = str(summarys[str(sample['relevant-id'][0])])
            user = 'Bart' if user_id == 1 else 'Lisa'
            info = 'Bart是一个性格外向、充满活力、情绪丰富且兴趣广泛的男孩，他对新事物充满好奇，喜欢参与各种活动，年龄在10-14岁左右。' if user_id == 1 else 'Lisa是一个性格外向、积极向上、多才多艺、善良且富有责任心的女孩，她重视家庭和朋友，情感丰富且敏感，并拥有较强的好胜心，年龄在6-11岁左右。'
            if len(test_turn) == 1:
            # order = random.sample(range(len(infer_res_path)), len(infer_res_path))

                context = ''.join(sample['dialogue'][:test_turn[0]]) + '<Assistant>:\n\n'
                ref_resp = sample['dialogue'][test_turn[0]].replace('<Assistant>:', '')
                prompt = ''
                prompt += ('\n##对话信息##\n历史事件P:\n{}\n').format(golden_summary)
                prompt += '当前对话时间: 2024-07-15\n'
                prompt += context
                
                response1='##候选回答 (a): {}\n'.format(infer_res[i*2][k])
                response2='##候选回答 (b): {}\n'.format(infer_res[i*2+1][k])

                k += 1

            elif len(test_turn) == 2:
                # order = random.sample(range(len(infer_res_path)), len(infer_res_path))
                t = test_turn[0]
                context = ''.join(sample['dialogue'][:t]) + '<Assistant>:\n\n'
                ref_resp_1 = sample['dialogue'][t].replace('<Assistant>:', '')
                ref_resp_2 = sample['dialogue'][t + 2].replace('<Assistant>:', '')
                ref_ans = sample['dialogue'][t + 1].replace(user_name[user_id - 1], '')
                prompt = ''

                prompt += ('\n##对话信息##\n历史事件P:\n{}\n').format(golden_summary)
                prompt += '当前对话时间: 2024-07-15\n'
                prompt += context
                
                response1= '##候选回答 (a)-1: {}\n'.format(infer_res[i*2][k])
                response2='##候选回答 (b)-1: {}\n'.format(infer_res[i*2+1][k])
                k += 1
                
                response1 += '\n##参考回答-1:{}'.format(ref_resp_1)
                response1 += '##参考用户回答-2:{}\n'.format(ref_ans)
                response2 += '\n##参考回答-1:{}'.format(ref_resp_1)
                response2 += '##参考用户回答-2:{}\n'.format(ref_ans)
                
                response1+= '##候选回答 (a)-2: {}\n'.format(infer_res[i*2][k])
                response2+='##候选回答 (b)-2: {}\n'.format(infer_res[i*2+1][k])
                k += 1
                
            else : #len(test_turn) == 3:
                # order = random.sample(range(len(infer_res_path)), len(infer_res_path))
                t = test_turn[0]
                context = ''.join(sample['dialogue'][:t]) + '<Assistant>:\n\n'
                ref_resp_1 = sample['dialogue'][t].replace('<Assistant>:', '')
                ref_resp_2 = sample['dialogue'][t + 2].replace('<Assistant>:', '')
                ref_resp_3 = sample['dialogue'][t + 4].replace('<Assistant>:', '')
                ref_ans = sample['dialogue'][t + 1].replace(user_name[user_id - 1], '')
                ref_ans_2 = sample['dialogue'][t + 3].replace(user_name[user_id - 1], '')

                prompt = ''

                prompt += ('\n##对话信息##\n历史事件P:\n{}\n').format(golden_summary)
                prompt += '当前对话时间: 2024-07-15\n'
                prompt += context
                
                response1= '##候选回答 (a)-1: {}\n'.format(infer_res[i*2][k])
                response2='##候选回答 (b)-1: {}\n'.format(infer_res[i*2+1][k])
                k += 1
                
                response1 += '\n##参考回答-1:{}'.format(ref_resp_1)
                response1 += '##参考用户回答-2:{}\n'.format(ref_ans)
                response2 += '\n##参考回答-1:{}'.format(ref_resp_1)
                response2 += '##参考用户回答-2:{}\n'.format(ref_ans)
                
                response1+= '##候选回答 (a)-2: {}\n'.format(infer_res[i*2][k])
                response2+='##候选回答 (b)-2: {}\n'.format(infer_res[i*2+1][k])
                k += 1
                response1 += '\n##参考回答-2:{}'.format(ref_resp_3)
                response1 += '##参考用户回答-3:{}\n'.format(ref_ans_2)
                response2 += '\n##参考回答-2:{}'.format(ref_resp_3)
                response2 += '##参考用户回答-3:{}\n'.format(ref_ans_2)

                response1+= '##候选回答 (a)-3: {}\n'.format(infer_res[i*2][k])
                response2+='##候选回答 (b)-3: {}\n'.format(infer_res[i*2+1][k])
                k += 1
                print(response1)
                print(response2)
            with open(anno_out_file, 'a+', encoding='utf-8') as file:
                json.dump({'model_name':model_name,'prompt': prompt,'model1':response1,'model2':response2,'id':sample['id']}, file, ensure_ascii=False)
                file.write('\n')
                if sample['id']==66:
                    print(prompt)
        # if n == 20:
        #     break
        # n += 1
    dials=read_json(anno_out_file)
    anno_out_file2=f'output/sbs_without_guide_0820_160_{model_name}.xlsx'
    data=[]
    res1=[]
    res2=[]
    ids=[]
    better=[]
    for s in dials:
        data.append(s['prompt'])
        model_name=s['model_name']
        res1.append(s['model1'])
        res2.append(s['model2'])
        ids.append(s['id'])
        better.append('')
    df=pd.DataFrame(data={'id':ids,'dialog':data,'response1':res1,'response2':res2,'better(a,b or t)':better},)
    df.to_excel(anno_out_file2,'sbs')
print('Done.')
            
            

    