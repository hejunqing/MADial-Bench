# encoding: utf8
from tqdm import tqdm

dialog_path='data/en/MADial-Bench-en-dialogue.json'
summary_path='data/en/MADial-Bench-en-memory.json'

infer_res_paths=[ # setting 3
    'output/en/setting3/setting3-gpt-4-turbo.json',
    'output/en/setting3/setting3-gpt-4o.json',
    'output/en/setting3/Llama-3.1-70B-Instruct_no_guideline_1010.json',
    'output/en/setting3/Llama-3.1-8B-Instruct_no_guideline_1010.json',
    'output/en/setting3/Smaug-34B_no_guideline.json'
    ]

import json
def load_json_line(filename):
    data=[]
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# load results
count = 0
with open(infer_res_paths[0], 'r', encoding='utf-8') as file:
    for line in file:
        count += 1
infer_res = [[] for _ in infer_res_paths]
for i in range(len(infer_res_paths)):
    with open(infer_res_paths[i], 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=count, ncols=66):
            sample = json.loads(line)
            infer_res[i].append(sample)


role_info = {'Bart':'Bart is an outgoing, energetic, and emotional boy with a wide range of interests. He is curious about new things and enjoys participating in various activities. He is around 10 to 14 years old.' ,
'Lisa':'Lisa is an outgoing, positive, multi-talented, kind, and responsible girl. She values family and friends, is emotionally rich and sensitive, and has a strong competitive spirit. She is around 6 to 11 years old.'}


pre_prompt="""You are an experienced, completely impartial, and emotionally intelligent expert judge. Given {user} and Assistant's character backgrounds, a historical event P related to {user}, the time the conversation takes place, a segment of dialogue between Assistant and {user}, and a candidate responses that will serve as Assistant's replies to {user}, you need to explain and score according to various scoring criteria.
The dialogue starts with <BOD> and ends with <EOD>.

##Character profile of {user}##:
    {info}

##Character profile of Assistant##:
    1. Outgoing, speaks enthusiastically and fluently.
    2. Prefers using praise and encouragement in conversations.
    3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
    4. Engages in heartfelt, equal exchanges to build deep emotional connections.
    5. Always uses a tone similar to talking with children—simple and witty.
    6. A virtual character, not capable of physical activities.

##Dialogue and historial memory##:
    {dialogue}

##Candidate Responses##:
{response}
"""

def extract_dialog(context):
    return 'Current conversation date:'+context.split('Current conversation date:')[-1]

def group_turns(dialog_path,file_out):
    data=load_json_line(dialog_path)
    k=0
    for idx,d in enumerate(data):
        candidates=[]
        test_turn=d['test-turn']
        user_id=d['user-id']
        user = 'Bart' if user_id == 1 else 'Lisa'
        info= role_info[user]
        if len(test_turn)==1:
            turn=test_turn[0]
            dd=extract_dialog(infer_res[0][k]['context'])
            cands=[f'Candidate {it+1}: '+m[k]['response'].split('<|eot_id|>')[0] for it,m in enumerate(infer_res)]
            cands.append(f'Candidate {len(cands)+1}: '+d['dialogue'][turn].replace('<Assistant>:',''))
            prompt=pre_prompt.format(user=user,info=info,dialogue=dd,response='\n'.join(cands))
            candidates.append(cands)
            k+=1
        else:
            prompt=""
            for i,turn in enumerate(test_turn):
                if i:
                    prompt+= 'Reference response: {}'.format(ref_ans.replace('<Assistant>: ',''))
                    prompt+='{}'.format(d['dialogue'][turn-1])
                    prompt+=f'\n##Candidate Response for turn-{i+1}##:\n'
                # get prompt and dialog from results
                cands=[f"Candidate {it+1}: "+m[k]['response'].split('<|eot_id|>')[0] for it,m in enumerate(infer_res)]
                cands.append(f'Candidate {len(cands)+1}: '+d['dialogue'][turn].replace('<Assistant>: ',''))    
                candidates.append(cands)
                if not i:
                    dd=extract_dialog(infer_res[0][k]['context'])
                    prompt+=pre_prompt.format(user=user,info=info,dialogue=dd,response='\n'.join(cands))
                else:
                    prompt+='\n'.join(cands)
                print(idx,turn,prompt)
                ref_ans = d['dialogue'][turn]
                k+=1
        
        scores=[[[],[],[],[],[]],
                   [[],[],[],[],[]],
                    [[],[],[],[],[]],
                    [[],[],[],[],[]],
                    [[],[],[],[],[]],
                    [[],[],[],[],[]]],

        with open(file_out, 'a+', encoding='utf-8') as file:
            json.dump({'prompt': prompt,'candidates':candidates,'user':user,
                       'scores':scores,'id':idx,"test_turns":test_turn,"keys":["naturalness","style coherence","memory injection","emotional improvement"]}, file, ensure_ascii=False)
            file.write('\n')
            if idx==14:
                print(prompt)
                print('='*20)
            if idx==66:
                print(prompt)
        # if n == 20:
        #     break
        # n += 1
    print('Done.')
    
if __name__=='__main__':

    group_turns(dialog_path,'annotation/en/setting3_prompt_candidates_turns.json')