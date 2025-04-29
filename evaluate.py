#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Time    :   2024/05/31 16:45:09
@Author  :   He Junqing 
@Version :   1.0
@Contact :   hejunqing@idea.edu.cn
@Desc    :   None
'''
import torch
from bert_score import score 
from process_ano import read_json
# from evaluation.eval_rouge_l import calculate_rouge_l
import os
import jieba
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def chinese_bleu(reference_text, candidate_text):
    # 使用 jieba 进行中文分词
    reference_tokens = list(jieba.cut(reference_text))
    candidate_tokens = list(jieba.cut(candidate_text))

    # 设置不同的 n-gram 权重
    weights_1 = (1.0, 0, 0, 0)  # BLEU-1
    weights_2 = (0.5, 0.5, 0, 0)  # BLEU-2
    weights_3 = (0.33, 0.33, 0.33, 0)  # BLEU-3
    weights_4 = (0.25, 0.25, 0.25, 0.25)  # BLEU-4

    # 计算 BLEU 分数
    bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_1, smoothing_function=SmoothingFunction().method1)
    bleu_2 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_2, smoothing_function=SmoothingFunction().method1)
    bleu_3 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_3, smoothing_function=SmoothingFunction().method1)
    bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_4, smoothing_function=SmoothingFunction().method1)

    return bleu_1, bleu_2, bleu_3, bleu_4

def english_bleu(reference_text, candidate_text):
    reference_tokens,candidate_tokens=reference_text.split(), candidate_text.split()
    # 设置不同的 n-gram 权重
    weights_1 = (1.0, 0, 0, 0)  # BLEU-1
    weights_2 = (0.5, 0.5, 0, 0)  # BLEU-2
    weights_3 = (0.33, 0.33, 0.33, 0)  # BLEU-3
    weights_4 = (0.25, 0.25, 0.25, 0.25)  # BLEU-4

    # 计算 BLEU 分数
    bleu_1 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_1, smoothing_function=SmoothingFunction().method1)
    bleu_2 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_2, smoothing_function=SmoothingFunction().method1)
    bleu_3 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_3, smoothing_function=SmoothingFunction().method1)
    bleu_4 = sentence_bleu([reference_tokens], candidate_tokens, weights=weights_4, smoothing_function=SmoothingFunction().method1)

    return bleu_1, bleu_2, bleu_3, bleu_4

def ngrams(tokens, n):
    """生成n-grams列表"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def distinct_n(text, n):
    """计算给定文本的Distinct-n值"""
    # 使用jieba进行中文分词
    tokens = list(jieba.cut(text))
    # 生成n-grams
    n_grams = ngrams(tokens, n)
    # 计算不同n-grams的数量
    unique_ngrams = len(set(n_grams))
    # 计算所有n-grams的数量
    total_ngrams = len(n_grams)
    # 计算Distinct-n值
    if total_ngrams == 0:
        return 0
    return unique_ngrams / total_ngrams

def calculate_rouge_l(reference, hypothesis):
    # 对中文文本进行分词
    reference_tokens = " ".join(jieba.cut(reference))
    hypothesis_tokens = " ".join(jieba.cut(hypothesis))
    
    # 创建Rouge对象
    rouge = Rouge()
    
    # 计算ROUGE-L
    scores = rouge.get_scores(hypothesis_tokens, reference_tokens)
    
    # 获取ROUGE-L分数
    rouge_l = scores[0]['rouge-l']['f']
    
    return rouge_l


lang='en'
infer_res_path=[
    'output/en/setting1/gpt4_no_guideline.json',
    'output/en/setting1/gpt-4o_no_guideline.json',
    'output/en/setting1/Meta-Llama-3-8B-Instruct_no_guideline.json',
    'output/en/setting1/Meta-Llama-3-70B-Instruct_no_guideline.json',
    'output/en/setting1/Smaug-34B-v0.1_no_guideline.json'
]

for path_name in infer_res_path:
    model_name=os.path.basename(path_name)
    data=read_json(path_name)
    candidates=[]
    references=[]
    rougel=[]
    dist1=[]
    dist2=[]
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    for d in data:
        candidates.append(d['response'])
        reference=d['reference-response'].lstrip('<Assistant>:')
        references.append(reference)
        if lang=='en':
            bleu_scores= english_bleu(reference,d['response'])
        else:
            bleu_scores = chinese_bleu(reference, d['response'])
        bleu1 += bleu_scores[0]
        bleu2 += bleu_scores[1]
        bleu3 += bleu_scores[2]
        bleu4 += bleu_scores[3]
        rougel.append(calculate_rouge_l(reference,d['response']))
        dist1.append(distinct_n(d['response'],1))
        dist2.append(distinct_n(d['response'],2))
    P, R, F1 = score(candidates, references, model_type="bert-base-chinese",num_layers=12, verbose=True)
    # P, R, F1 = score(candidates, references, model_type="/cognitive_comp/hejunqing/projects/pretrained_models/bge-large-zh-v1.5",num_layers=24, verbose=True)

# 打印结果

    print("model:",model_name)
    print('dist-1',sum(dist1)/len(dist1))
    print('dist-2',sum(dist2)/len(dist2))
    print('rouge-l',sum(rougel)/len(rougel))
    # print("Bertscore Precision:", P)
    # print("Bertscore Recall:", R)
    print("Bertscore F1 Score:", torch.mean(F1), F1.shape)
    print("BLEU-1: ", bleu1 / len(data))
    print("BLEU-2: ", bleu2 / len(data))
    print("BLEU-3: ", bleu3 / len(data))
    print("BLEU-4: ", bleu4 / len(data))


score_prompt="""给定两个角色的人设以及他们部分对话内容，回忆的历史事件P，以及回复的准则，判断最后一轮候选回复的好坏。
###角色描述：
姓名：{user_name}
特点：{user_}

姓名：Assistant
特点：
    1.是一位注重个人形象和社交礼仪的人，形象优雅，言谈流利；
    2.具有很强的社交能力和人际交往技巧，能够在对话中轻松切入与他人兴趣相关的话题，拥有很好的审美能力，能够对他人进行恰当的评价和称赞；        
    3.性格外向，善于交际，重视个人形象，乐于表达自己的看法；        
    4.说话风格热情而略带赞赏，经常会用夸奖和鼓励的话语来与他人交流，并倾向于使用比喻和赞扬的语句来表达自己的观点；
    5.话语温暖亲切，能够和人贴心交谈，建立深入的情感链接；
    6.虚拟人物，不具备物质活动能力。
    7.对话内容自然、口语化；
    8.不会对孩子说教；
    9.以和孩子交流的口吻说话，简洁风趣；

###回复的准则：
{guideline}

对话的形式：
<角色>：角色回复

###对话内容：
{dialog}

回复好坏的判断标准：(得分为0，1，2。0表示不符合要求，1表示符合一个要求，2表示全都符合)
1.对话连贯、自然。
2.口语化、和小孩子的口吻对话。
3.角色一致性，和Assistant的特点是否符合。
4.回复是否引入了正确的记忆，引入的内容是否恰当、具体。
5.回复是否符合对话的准则。是否正确对应对话准则中的情况，是否完全遵循对应的准则回复。
6.能否理解对方的处境、进行共情，并提升对方的情绪。如果对方是正向情绪，则维持或提升；如果对方是负向情绪，则进行安抚或给出建议。
"""

