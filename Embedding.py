import numpy as np
from transformers import AutoModel
from numpy.linalg import norm
from tqdm import tqdm
import json
import heapq
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from FlagEmbedding import BGEM3FlagModel
from copy import copy
import openai


openai.api_key = "xxx"
model_name = 'text-embedding-3-large'

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
data_path='data/zh/Membench-zh-dialogue-final.json'
summary_path='data/Membench-zh-memory-final.json'

def calculate_similarity(embedding_path):
    embedding_dialogue, embedding_summary = [], []
    count = 0
    with open(embedding_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
    with open(embedding_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=count, ncols=120):
            sample = json.loads(line)
            embedding_dialogue.append(sample['dialogue'])
            embedding_summary.append(sample['summary'])

    embedding_dialogue = np.array(embedding_dialogue)
    embedding_summary = np.array(embedding_summary)
    scores = []
    for em_dia in embedding_dialogue:
        sims = []
        for em_sum in embedding_summary:
            # sims.append(cos_sim(em_dia, em_sum)) # Jina
            sims.append(em_dia @ em_sum.T) # Acge / Bge / Stella
        scores.append(sims)
    res_top_1, res_top_5, res_top_10, res_top_20 = [], [], [], []
    for s in scores:
        res_top_1.append(s.index(max(s)))
        largest_five = heapq.nlargest(5, s)
        largest_ten = heapq.nlargest(10, s)
        largest_twenty = heapq.nlargest(20, s)
        res_top_5.append([s.index(element) for element in largest_five])
        res_top_10.append([s.index(element) for element in largest_ten])
        res_top_20.append([s.index(element) for element in largest_twenty])

    count_top_1, count_top_5, count_top_10, count_top_20 = 0, 0, 0, 0
    for i in range(len(res_top_1)):
        if res_top_1[i] == i:
            count_top_1 += 1
        if i in res_top_5[i]:
            count_top_5 += 1
        if i in res_top_10[i]:
            count_top_10 += 1
        if i in res_top_20[i]:
            count_top_20 += 1
    acc_1 = count_top_1 / count
    acc_5 = count_top_5 / count
    acc_10 = count_top_10 / count
    acc_20 = count_top_20 / count
    return acc_1, acc_5, acc_10, acc_20

def generate_embedding(model, data_path, embedding_path, entire):
    count = 0
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=count, ncols=120):
            sample = json.loads(line)
            summary = sample['summary']
            summary.pop('summary-id')
            summary['user'] = 'Bart' if summary['user-id'] == 1 else 'Lisa'
            summary.pop('user-id')

            dia = '<Time>: 2024-06-16\n'
            dialogue = sample['dialogue']

            if entire:
                for x in dialogue:  # 整段
                    dia += x
            else:
                test_turn = sample['test-turn'][0]  # 只测试到第一次 test 回复之前
                for i in range(len(dialogue)):
                    if i < test_turn:
                        dia += dialogue[i]

            embedding_sum = model.encode(str(summary), normalize_embeddings=True)
            embedding_dia = model.encode(dia, normalize_embeddings=True)

            with open(embedding_path, 'a+', encoding='utf-8') as file:
                json.dump({
                    "summary": embedding_sum.tolist(),
                    "dialogue": embedding_dia.tolist()
                    }, file, ensure_ascii=False)
                file.write('\n')
                
def gen_summary_embedding(filename,file_out,model_name=''):
    if model_name in ['BGE_M3_dense','BGE_M3_colbert']:
        model = BGEM3FlagModel('./pretrained_models/bge-m3', use_fp16=True)
    elif model_name =='Acge':
        model = SentenceTransformer('./pretrained_models/acge_text_embedding')
    elif model_name=='Stella':
        model = SentenceTransformer('./pretrained_models/stella-large-zh-v3-1792d')
    elif model_name=='Dmeta':
        model= SentenceTransformer('./pretrained_models/Dmeta-embedding-zh')
    embeddings=[]
    with open(filename,'r',encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            d=json.loads(line)
            for k,v in d.items():
                s=copy(v)
                uid=s.pop('user-id')
                id=s.pop('id')
                s.pop('time')
                summary=s
                if model_name=='BGE_M3_dense':
                    # 
                    embedding_sum= model.encode(str(summary))['dense_vecs'].tolist()
                elif model_name=='BGE_M3_colbert':
                    embedding_sum = model.encode(str(summary), return_colbert_vecs=True)['colbert_vecs'].tolist()
                elif model_name=='text-embedding-3-large':
                    embedding_sum=openai.Embedding.create(model=model_name, input=str(summary))['data'][0]['embedding']
                else:
                    embedding_sum=model.encode(str(s),normalize_embeddings=True).tolist()
                record={'id':id,'embeddings':embedding_sum,'user-id':uid,'event':v['event'],'emotion':v['emotion'],'scene':v['scene']}
                embeddings.append(record)
                with open(file_out, 'a+', encoding='utf-8') as file:
                    json.dump(record, file, ensure_ascii=False)
                    file.write('\n')
    print(len(embeddings),f'summary embeddings written with {model_name}' )
    
def gen_dialog_embedding(filename,file_out,model_name=''):
    if model_name in ['BGE_M3_dense','BGE_M3_colbert']:
        model = BGEM3FlagModel('./pretrained_models/bge-m3', use_fp16=True)
    elif model_name =='Acge':
        model = SentenceTransformer('./pretrained_models/acge_text_embedding')
    elif model_name=='Stella':
        model = SentenceTransformer('./pretrained_models/stella-large-zh-v3-1792d')
    elif model_name=='Dmeta':
        model= SentenceTransformer('./pretrained_models/Dmeta-embedding-zh')
    embeddings=[]
    with open(filename,'r',encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            d=json.loads(line)
            t=d['test-turn'][0]
            dial=''.join(d['dialogue'][:t])
            id=d['id']
            if dial:
                if model_name=='BGE_M3_dense':
                    # 
                    embedding_sum= model.encode(str(dial))['dense_vecs'].tolist()
                elif model_name=='BGE_M3_colbert':
                    embedding_sum = model.encode(str(dial), return_colbert_vecs=True)['colbert_vecs'].tolist()
                elif model_name=='text-embedding-3-large':
                    embedding_sum=openai.Embedding.create(model=model_name, input=str(dial))['data'][0]['embedding']
                else:
                    embedding_sum=model.encode(str(dial),normalize_embeddings=True).tolist()
                record={'id':id,'embeddings':embedding_sum,'dialogue':dial}
                embeddings.append(record)
                with open(file_out, 'a+', encoding='utf-8') as file:
                    json.dump(record, file, ensure_ascii=False)
                    file.write('\n')
    print(len(embeddings),f'dialog embeddings written with {model_name}' )


if __name__=='__main__':
    import sys
    if sys.argv[1]=='dialog':
        gen_dialog_embedding(data_path,'embeddings/openai_zh_dialog_embeddings.json',model_name=model_name)
        gen_dialog_embedding(data_path,'embeddings/BGE_M3_dense_zh_dialog_embeddings.json',model_name='BGE_M3_dense')
        gen_dialog_embedding(data_path,'embeddings/BGE_M3_colbert_zh_dialog_embeddings.json',model_name='BGE_M3_colbert')
        gen_dialog_embedding(data_path,'embeddings/Acge_zh_dialog_embeddings.json',model_name='Acge')
        gen_dialog_embedding(data_path,'embeddings/Stella_zh_dialog_embeddings.json',model_name='Stella')
        gen_dialog_embedding(data_path,'embeddings/Dmeta_zh_dialog_embeddings.json',model_name='Dmeta')
    else:
        gen_summary_embedding(summary_path,'embeddings/openai_zh_summary_embeddings.json',model_name=model_name)
        gen_summary_embedding(summary_path,'embeddings/BGE_M3_dense_zh_summary_embeddings.json',model_name='BGE_M3_dense')
        gen_summary_embedding(summary_path,'embeddings/BGE_M3_colbert_zh_summary_embeddings.json',model_name='BGE_M3_colbert')
        gen_summary_embedding(summary_path,'embeddings/Acge_zh_summary_embeddings.json',model_name='Acge')
        gen_summary_embedding(summary_path,'embeddings/Stella_zh_summary_embeddings.json',model_name='Stella')
        gen_summary_embedding(summary_path,'embeddings/Dmeta_zh_summary_embeddings.json',model_name='Dmeta')
        



















