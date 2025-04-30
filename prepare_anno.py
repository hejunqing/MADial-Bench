# encoding:utf8
import os
import json
import pandas as pd
from make_annotion_candidates import load_json_line
import random

def main(data_file,excel_file,lang):
    for i in range(2,3):
        random.seed(42)
        filename=data_file.format(lang,i+1)
        out_file=excel_file.format(lang,i+1)
        res=load_json_line(filename)
        random.shuffle(res)
        dialog_list=[]
        candidate_1=[]
        candidate_2=[]
        candidate_3=[]
        candidate_4=[]
        candidate_5=[]
        candidate_6=[]
        most_intimate=[]
        ids=[]
        candidates=[]
        prompts=[]
        headers=['id','prompt','dialog','responses','candidate 1','candidate 2','candidate 3','candidate 4','candidate 5','candidate 6','most intimate one']
        for s in res[:20]:
            ids.append(s['id'])
            prompt,dialog=s['prompt'].split('##Dialogue and historial memory##:')
            dialog,candidate_response=dialog.split('##Candidate Responses##:')
            dialog_list.append('##Dialogue and historial memory##:'+dialog)
            prompts.append(prompt)
            candidates.append('##Candidate Responses##:'+candidate_response)
            candidate_1.append({"naturalness":[],"style coherence":[],"memory injection":[],"emotional improvement":[],"ES skills":[]})
            candidate_2.append({"naturalness":[],"style coherence":[],"memory injection":[],"emotional improvement":[],"ES skills":[]})
            candidate_3.append({"naturalness":[],"style coherence":[],"memory injection":[],"emotional improvement":[],"ES skills":[]})
            candidate_4.append({"naturalness":[],"style coherence":[],"memory injection":[],"emotional improvement":[],"ES skills":[]})
            candidate_5.append({"naturalness":[],"style coherence":[],"memory injection":[],"emotional improvement":[],"ES skills":[]})
            candidate_6.append({"naturalness":[],"style coherence":[],"memory injection":[],"emotional improvement":[],"ES skills":[]})
            most_intimate.append('')    
        df=pd.DataFrame(data={'id':ids,'prompts':prompts,'dialog':dialog_list,'responses':candidates,'candidate 1':candidate_1,'candidate 2':candidate_2,'candidate 3':candidate_3,'candidate 4':candidate_4,'candidate 5':candidate_5,'candidate 6':candidate_6,'most intimate one':most_intimate})
        df.to_excel(out_file,sheet_name='annotation')
        print('setting {} excel done'.format(i+1))
    print('Done')

lang='en' # 'ch'
main('annotation/{}/setting{}_prompt_candidates_turns.json','output/{}/setting{}_anno_new.xlsx',lang)