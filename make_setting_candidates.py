#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   make_t2_candidates.py
@Time    :   2024/07/03 21:53:46
@Author  :   He Junqing 
@Version :   1.0
@Contact :   hejunqing@idea.edu.cn
@Desc    :   None
'''

import json
from tqdm import tqdm 
from embedding_top_20_new import read_json

# neg_path_zh='data/embeddings/ch/openai_zh_top_20_ids.json' 
# dialog_path='data/MADial-Bench-zh-dialogue.json'
# summary_path='data/MADial-Bench-zh-memory.json'
neg_path_zh='embeddings/en/openai_top_20_bottom.json'
dialog_path='data/en/MADial-Bench-en-dialogue.json'
summary_path='data/en/MADial-Bench-en-memory.json'

def read_summary():
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
    return summary
    
def gen_setting2(dialog_file,summary,neg_path,out_file):
    import random
    random.seed(1991)
    data=read_json(neg_path)
    dd=read_json(dialog_file)
    fw=open(out_file,'w',encoding='utf8')
    for d1,d2 in zip(data,dd):
        pred_ids=d1['top_20_ids']
        groud_truth=d1['relevant_id']
        neg=[]
        for e in pred_ids:
            if e not in groud_truth:
                neg.append(e)
            if len(neg)==4:
                break
        his=neg+[groud_truth[0]]
        random.shuffle(his)
        memory=[summary[str(i)] for i in his]
        d2['memory']=memory
        d2['memory_ids']=his
        fw.write(json.dumps(d2,ensure_ascii=False))
        fw.write('\n')
    fw.close()
    print('Done')    
    
def gen_setting3(dialog_file,summary,neg_path,out_file):
    data=read_json(neg_path)
    dd=read_json(dialog_file)
    fw=open(out_file,'w',encoding='utf8')
    for d1,d2 in zip(data,dd):
        pred_ids=d1['top_20_ids']
        groud_truth=d1['relevant_id']
        his=pred_ids[:5]
        # random.shuffle(his)
        memory=[summary[str(i)] for i in his]
        d2['memory']=memory
        d2['memory_ids']=his
        fw.write(json.dumps(d2,ensure_ascii=False))
        fw.write('\n')
    fw.close()
    print('Done') 

summary=read_summary()
print(summary.keys())

gen_setting3(dialog_path,summary,neg_path_zh,'data/en/MADial-Bench-en-dialogue-setting3.json')
gen_setting2(dialog_path,summary,neg_path_zh,'data/en/MADial-Bench-en-dialogue-setting2.json')
