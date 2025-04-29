import numpy as np
from transformers import AutoModel
from numpy.linalg import norm
from tqdm import tqdm
import json
import heapq
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from FlagEmbedding import BGEM3FlagModel
import os
# from sentence_transformers.util import cos_sim # for gte

# cos_sim = lambda a, b: (a @ b.T) / (norm(a)*norm(b)) # for Jina
cos_sim = lambda a, b: (a @ b.T)

# 注意中英文
# summary_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/Mem-bench-en-summary.json'

def load_summary_embeddings(file_path):
    with open(file_path,'r',encoding='utf8') as f:
        embeddings ={}
        ids={}
        max_length=0
        # pad={}
        for line in f.readlines():
            data=json.loads(line)
            # print(len(data['embeddings']))
            ll=len(data['embeddings'])
            if isinstance(data['embeddings'][0],list):
                wl=len(data['embeddings'][0])
                print('shape',ll,wl)

            if data['user-id'] not in embeddings:
                embeddings[data['user-id']]=[data['embeddings']]
                ids[data['user-id']]=[int(data['id'])]
                # pad[data['user-id']]=[0]
            else:
                embeddings[data['user-id']].append(data['embeddings'])
                ids[data['user-id']].append(int(data['id']))

            if ll>max_length:
                print(ll)
                p=ll-max_length
                max_length=ll
            #     pad[data['user-id']]=[i+p for i in pad[data['user-id']]]
            #     if len(pad[data['user-id']])>1:
            #         pad[data['user-id']].append(0)
            # else:
            #     p=max_length-ll
            #     pad[data['user-id']].append(p)
            print(max_length)
        return embeddings,ids,max_length
    
def pad_summary_embeddings(embeddings,max_length):
    new_embeddings=[]
    print('padding...')
    for patch in embeddings:
        ll=len(patch)
        p=max_length-ll
        print(ll,p)
        after=np.pad(patch,((0,p),(0,0)),'constant')
        print(after.shape)
        new_embeddings.append(after)
        
        # print('embeddings:',embeddings)
        # print(len(new_embeddings),len(new_embeddings[0]))
    return new_embeddings
        

def load_dialog_embeddings(file_path):
    with open(file_path,'r',encoding='utf8') as f:
        data=[]
        for line in f.readlines():
            d=json.loads(line)
            data.append(d)
        return data
    
def read_json(filename):
    dd=[]
    with open(filename,'r',encoding='utf8') as f:
        for line in f.readlines():
            dd.append(json.loads(line))
    return dd
                
def main(out_dir):
    for model in ['openai']:#['Acge','BGE_M3_dense','Dmeta','Stella']:
        summary_embeddings,summary_ids,pad=load_summary_embeddings('data/embeddings/'+model+'_zh_summary_embeddings.json')
        dialog_embeddings=load_dialog_embeddings('data/embeddings/'+model+'_zh_dialog_embeddings.json')
        dialogues=read_json('data/Membench-zh-dialogue-fix5.json')
        out_file=os.path.join(out_dir,'data/embeddings/'+model+'_zh_top_20_ids.json')
        fw=open(out_file,'w',encoding='utf8')
        for line,dd in zip(dialog_embeddings,dialogues):
            emd=np.array(line['embeddings'])
            ems=np.array(summary_embeddings[dd['user-id']])
            ids=np.array(summary_ids[dd['user-id']]).reshape(-1,1)
            print('ids:',ids)
            similarity=cos_sim(ems,emd.reshape(1,-1))
            print(similarity.shape)
            top_k_indices = np.argsort(similarity.reshape(1,-1)).flatten()[-20:][::-1].tolist()
            print('top_k_indices:',top_k_indices)
            top_k_ids=ids[top_k_indices].flatten().tolist()
            print(top_k_ids)
            # top_k_ids = [int(x) for x in top_k_ids]
            
            out={'id':line['id'],'user-id':dd['user-id'],'top_20_ids':top_k_ids,'relevant_id':dd['relevant-id'],'dialog_history':line['dialogue'],'dialogue':dd['dialogue'],'test-turn':dd['test-turn']}
            fw.write(json.dumps(out,ensure_ascii=False))
            fw.write('\n')
            
def main2(out_dir):
    for model in ['BGE_M3_colbert']:
        summary_embeddings,summary_ids,max_length=load_summary_embeddings('data/embeddings/ch/'+model+'_zh_memory_embeddings.json')
        dialog_embeddings=load_dialog_embeddings('data/embeddings/ch/'+model+'_zh_dialog_embeddings.json')
        dialogues=read_json('data/MADial-Bench-zh-dialogue.json')
        out_file=os.path.join(out_dir,'data/embeddings/ch/'+model+'_zh_top_20_ids.json')
        fw=open(out_file,'a+',encoding='utf8')
        for line,dd in zip(dialog_embeddings,dialogues):
            emd=np.array(line['embeddings'])
            ems=np.array(pad_summary_embeddings(summary_embeddings[dd['user-id']],max_length))
            ids=np.array(summary_ids[dd['user-id']]).reshape(-1,1)
            print('ids:',ids)
            similarity=cos_sim(ems,emd)
            print(similarity.shape)
            sum_max=np.sum(np.max(similarity,axis=1,keepdims=False),axis=-1,keepdims=True)
            top_k_indices = np.argsort(sum_max.reshape(1,-1)).flatten()[-20:][::-1].tolist()
            print('top_k_indices:',top_k_indices)
            top_k_ids=ids[top_k_indices].flatten().tolist()
            print(top_k_ids)
            # top_k_ids = [int(x) for x in top_k_ids]
            
            out={'id':line['id'],'user-id':dd['user-id'],'top_20_ids':top_k_ids,'relevant_id':dd['relevant-id'],'dialog_history':line['dialogue'],'dialogue':dd['dialogue'],'test-turn':dd['test-turn']}
            fw.write(json.dumps(out,ensure_ascii=False))
            fw.write('\n')
            
if __name__=='__main__':

    main('.')
            
            




# # Jina zh
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/jina-embeddings-v2-base-zh'
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/jina.json'
# emb_list = []
# for s in summary:
#     emb_list.append(model.encode(str(s), normalize_embeddings=True).tolist())
# with open(embedding_path, 'a+', encoding='utf-8') as file:
#     json.dump({
#         "summary": emb_list,
#         "ids": id_list
#     }, file, ensure_ascii=False)
#     file.write('\n')
# print('Done.')
#
# # Acge zh
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/acge_text_embedding'
# model = SentenceTransformer(model_path)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/acge.json'
# emb_list = []
# for s in summary:
#     emb_list.append(model.encode(str(s), normalize_embeddings=True).tolist())
# with open(embedding_path, 'a+', encoding='utf-8') as file:
#     json.dump({
#         "summary": emb_list,
#         "ids": id_list
#     }, file, ensure_ascii=False)
#     file.write('\n')
# print('Done.')
#
# # Stella zh
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/stella-large-zh-v3-1792d'
# model = SentenceTransformer(model_path)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/stella.json'
# emb_list = []
# for s in summary:
#     emb_list.append(model.encode(str(s), normalize_embeddings=True).tolist())
# with open(embedding_path, 'a+', encoding='utf-8') as file:
#     json.dump({
#         "summary": emb_list,
#         "ids": id_list
#     }, file, ensure_ascii=False)
#     file.write('\n')
# print('Done.')
#
# # Bge zh
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/bge-large-zh-v1.5'
# model = SentenceTransformer(model_path)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/bge.json'
# emb_list = []
# for s in summary:
#     emb_list.append(model.encode(str(s), normalize_embeddings=True).tolist())
# with open(embedding_path, 'a+', encoding='utf-8') as file:
#     json.dump({
#         "summary": emb_list,
#         "ids": id_list
#     }, file, ensure_ascii=False)
#     file.write('\n')
# print('Done.')
#
# # BGE M3  zh / en
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/bge-m3'
# model = BGEM3FlagModel(model_path, use_fp16=True)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/bge_m3.json'
# emb_list = []
# for s in tqdm(summary, total=160, ncols=66):
#     emb_list.append(model.encode(str(s))['dense_vecs'].tolist())
# with open(embedding_path, 'a+', encoding='utf-8') as file:
#     json.dump({
#         "summary": emb_list,
#         "ids": id_list
#     }, file, ensure_ascii=False)
#     file.write('\n')
# print('Done.')

# # Dmeta zh
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/Dmeta-embedding-zh'
# model = SentenceTransformer(model_path)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/dmeta.json'
# emb_list = []
# for s in summary:
#     emb_list.append(model.encode(str(s), normalize_embeddings=True).tolist())
# with open(embedding_path, 'a+', encoding='utf-8') as file:
#     json.dump({
#         "summary": emb_list,
#         "ids": id_list
#     }, file, ensure_ascii=False)
#     file.write('\n')
# print('Done.')


# # Jina en
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/jina-embeddings-v2-base-en'
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/jina.json'
# emb_list = []
# for s in summary:
#     emb_list.append(model.encode(str(s), normalize_embeddings=True).tolist())
# with open(embedding_path, 'a+', encoding='utf-8') as file:
#     json.dump({
#         "summary": emb_list,
#         "ids": id_list
#     }, file, ensure_ascii=False)
#     file.write('\n')
# print('Done.')

# # gte en
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/gte-large-en-v1.5/gte-large-en-v1.5'
# model = SentenceTransformer(model_path, trust_remote_code=True)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/gte.json'
# emb_list = []
# for s in summary:
#     emb_list.append(model.encode(str(s)).tolist())
# with open(embedding_path, 'a+', encoding='utf-8') as file:
#     json.dump({
#         "summary": emb_list,
#         "ids": id_list
#     }, file, ensure_ascii=False)
#     file.write('\n')
# print('Done.')

# # OpenAI en (zh)
# import openai
# openai.api_key = "xxx"
# model = 'text-embedding-3-large'
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/openai.json'
# emb_list = []
# for s in summary:
#     emb_list.append(openai.Embedding.create(model=model, input=str(s))['data'][0]['embedding'])
# with open(embedding_path, 'a+', encoding='utf-8') as file:
#     json.dump({
#         "summary": emb_list,
#         "ids": id_list
#     }, file, ensure_ascii=False)
#     file.write('\n')
# print('Done.')



# 计算 top 20     OpenAI 另取 top -10
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/jina-embeddings-v2-base-en' # jina
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/jina.json'
#
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/gte-large-en-v1.5/gte-large-en-v1.5'
# model = SentenceTransformer(model_path, trust_remote_code=True)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/gte.json'
#
# model_path = '/cognitive_comp/hejunqing/projects/pretrained_models/bge-m3'
# model = BGEM3FlagModel(model_path, use_fp16=True)
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/bge_m3.json'


# OpanAI en
# import openai
# openai.api_key = "sk-proj-CsM3ltN9MnXZrDsdKANJT3BlbkFJW3x3aPqaG4yopp7Toqy5"
# model = 'text-embedding-3-large'
# embedding_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/openai.json'
#
# with open(embedding_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         sample = json.loads(line)
#         sum_emb = sample['summary']
#         sum_ids = sample['ids']
# bart_emb, bart_ids = sum_emb[:76], sum_ids[:76]
# lisa_emb, lisa_ids = sum_emb[76:], sum_ids[76:]
# bart_emb, lisa_emb, bart_ids, lisa_ids = np.array(bart_emb), np.array(lisa_emb), np.array(bart_ids), np.array(lisa_ids)
# dialogue_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/Mem-bench-en-dialogue.json'
# count = 0
# with open(dialogue_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         count += 1
# with open(dialogue_path, 'r', encoding='utf-8') as file:
#     for line in tqdm(file, total=count, ncols=66):
#         sample = json.loads(line)
#         test_turn = sample['test-turn'][0]
#         dialogue = sample['dialogue']
#         user_id = sample['user-id']
#         part_dialogue = ''.join(dialogue[:test_turn])
#         # part_emb = model.encode(part_dialogue, normalize_embeddings=True)
#         # part_emb = model.encode(part_dialogue) # for gte
#         # part_emb = model.encode(part_dialogue)['dense_vecs'] # for BGE M3
#         part_emb = np.array(openai.Embedding.create(model=model, input=part_dialogue)['data'][0]['embedding'])
#         if user_id == 1:
#             similarity = cos_sim(bart_emb, part_emb.reshape(1, -1)).flatten()
#             top_k_indices = np.argsort(similarity)[-20:][::-1].tolist()
#             top_k_ids = bart_ids[top_k_indices].tolist()
#             top_k_ids = [int(x) for x in top_k_ids]
#
#             bottom_k_indices = np.argsort(similarity)[:10].tolist()
#             bottom_k_ids = bart_ids[bottom_k_indices].tolist()
#             bottom_k_ids = [int(x) for x in bottom_k_ids]
#         else:
#             similarity = cos_sim(lisa_emb, part_emb.reshape(1, -1)).flatten()
#             top_k_indices = np.argsort(similarity)[-20:][::-1].tolist()
#             top_k_ids = lisa_ids[top_k_indices].tolist()
#             top_k_ids = [int(x) for x in top_k_ids]
#
#             bottom_k_indices = np.argsort(similarity)[:10].tolist()
#             bottom_k_ids = lisa_ids[bottom_k_indices].tolist()
#             bottom_k_ids = [int(x) for x in bottom_k_ids]
#
#         sample.pop('dialogue')
#
#         sample['top-20-ids'] = top_k_ids
#         sample['bottom-10-ids'] = bottom_k_ids
#
#         with open('/cognitive_comp/zhuliang/data/Mem-bench/en/openai_top_20_bottom.json', 'a+', encoding='utf-8') as file:
#             json.dump(sample, file, ensure_ascii=False)
#             file.write('\n')
# print('Done.')


# OpenAI zh
# import openai
# openai.api_key = "sk-proj-OaAH3F4imVa6O9wgEoiMT3BlbkFJ5TWIALPqAxKCPZFg3ROD"
# model = 'text-embedding-3-large'
# embedding_path = 'output/embedding_top20_zh/openai.json'

# with open(embedding_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         sample = json.loads(line)
#         sum_emb = sample['summary']
#         sum_ids = sample['ids']
# bart_emb, bart_ids = sum_emb[:76], sum_ids[:76]
# lisa_emb, lisa_ids = sum_emb[76:], sum_ids[76:]
# bart_emb, lisa_emb, bart_ids, lisa_ids = np.array(bart_emb), np.array(lisa_emb), np.array(bart_ids), np.array(lisa_ids)
# dialogue_path = 'data/Membench-zh-dialogue-fix5.json'
# count = 0
# with open(dialogue_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         count += 1
# with open(dialogue_path, 'r', encoding='utf-8') as file:
#     for line in tqdm(file, total=count, ncols=66):
#         sample = json.loads(line)
#         test_turn = sample['test-turn'][0]
#         dialogue = sample['dialogue']
#         user_id = sample['user-id']
#         part_dialogue = ''.join(dialogue[:test_turn])
#         # part_emb = model.encode(part_dialogue, normalize_embeddings=True)
#         # part_emb = model.encode(part_dialogue) # for gte
#         # part_emb = model.encode(part_dialogue)['dense_vecs'] # for BGE M3
#         part_emb = np.array(openai.Embedding.create(model=model, input=part_dialogue)['data'][0]['embedding'])
#         if user_id == 1:
#             similarity = cos_sim(bart_emb, part_emb.reshape(1, -1)).flatten()
#             top_k_indices = np.argsort(similarity)[-20:][::-1].tolist()
#             top_k_ids = bart_ids[top_k_indices].tolist()
#             top_k_ids = [int(x) for x in top_k_ids]

#             bottom_k_indices = np.argsort(similarity)[:10].tolist()
#             bottom_k_ids = bart_ids[bottom_k_indices].tolist()
#             bottom_k_ids = [int(x) for x in bottom_k_ids]
#         else:
#             similarity = cos_sim(lisa_emb, part_emb.reshape(1, -1)).flatten()
#             top_k_indices = np.argsort(similarity)[-20:][::-1].tolist()
#             top_k_ids = lisa_ids[top_k_indices].tolist()
#             top_k_ids = [int(x) for x in top_k_ids]

#             bottom_k_indices = np.argsort(similarity)[:10].tolist()
#             bottom_k_ids = lisa_ids[bottom_k_indices].tolist()
#             bottom_k_ids = [int(x) for x in bottom_k_ids]

#         sample.pop('dialogue')

#         sample['top-20-ids'] = top_k_ids
#         sample['bottom-10-ids'] = bottom_k_ids

#         with open('/cognitive_comp/zhuliang/data/Mem-bench/openai_top_20_bottom.json', 'a+', encoding='utf-8') as file:
#             json.dump(sample, file, ensure_ascii=False)
#             file.write('\n')
# print('Done.')



















