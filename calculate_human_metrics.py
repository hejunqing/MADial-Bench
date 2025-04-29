# encoding: utf-8
import os
import json
from claude_endpoint import load_json_line
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
 
def get_average_results(filename,out_file):
    data=load_json_line(filename)
    res={}
    for line in tqdm(data):
        line=line['result']
        for result in line['scores']:
            candidate_id=str(result['candidate id']).replace('Candidate ','')
            if 'turn' in candidate_id:
                continue
            if candidate_id not in res:
                res[candidate_id]=defaultdict(list)
                # print(candidate_id,line)
            for k,v in result.items():
                if k=='candidate id' or k=='reason':
                    continue
                res[candidate_id][k].append(sum(v))
    print(res.keys())
    for k,v in res.items(): # candidate, dict
        for k1,v1 in v.items(): # metric, list
            avg=sum(v1)/len(v1)
            res[k][k1]=avg
    print(res)
    with open(out_file,'w',encoding='utf8') as f:
        json.dump(res,f,ensure_ascii=False)
    print('Avg scores Done')
    return res

def main(prefix_name,excel_path):
    for i in range(3):
        filename=prefix_name.format(i+1)
        out_file=excel_path.format(i+1)
        avg_file='avg_scores_setting{}_1012.json'.format(i+1)
        res=get_average_results(filename,avg_file)
        res_list=[[] for i in range(len(res.keys()))]
        print(res_list)
        header=[]
        for k,v in res.items():
            for dim,score in v.items():
                res_list[int(k)-1].append(score)
                if dim not in header:
                    header.append(dim)
        df=pd.DataFrame(res_list,columns=header)
        df.to_excel(out_file,sheet_name='setting{}'.format(i+1))
    print('Excel Done')

def add_column(prefix_name,excel_path):
    for i in range(1,3):
        filename=prefix_name.format(i+1)
        out_file=excel_path.format(i+1)
        avg_file='avg_scores_es_setting{}_1012.json'.format(i+1)
        res=get_average_results(filename,avg_file)
        res_list=[[] for i in range(len(res.keys()))]
        df=pd.read_excel(out_file,sheet_name='setting{}'.format(i+1))
        header=''
        for k,v in res.items():
            for dim,score in v.items():
                res_list[int(k)-1].append(score)
                header=dim
        values=[]
        for j in range(len(res_list)):
            values.append(res_list[j][0])
        
        # col_name=df.columns.tolist()
        # col_name.insert(-2,header)
        # wb=df.reindex(columns=col_name)
        df=pd.concat([df,pd.Series(values,name=header)],axis=1)
        df.to_excel(out_file,sheet_name='setting{}_es'.format(i+1))
        print('Add column Done')
        

if __name__=='__main__':
    # get_average_results('/Users/nicolehe/Desktop/mysea/projects/membench/output/scores_setting1_1011_last.json','avg_scores_setting1_1012.json')
    # main('output/scores_setting{}_1011_last.json','avg_scores_setting{}_1012.xlsx')
    add_column('/Users/nicolehe/Desktop/mysea/projects/membench/output/setting{}_es_scores.json','avg_scores_setting{}_1012.xlsx')