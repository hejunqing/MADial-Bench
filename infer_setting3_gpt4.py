import os
import json
from tqdm import tqdm
from openai import OpenAI

# openai.api_key = "sk-proj-RMBB0HYrEYHp0XHxIPcBT3BlbkFJHetoNwH9JOHzAnHMRf6c"
model = "gpt-4-turbo-2024-04-09"
client = OpenAI(api_key='sk-proj-OaAH3F4imVa6O9wgEoiMT3BlbkFJ5TWIALPqAxKCPZFg3ROD')

def get_completion(prompt, model=model):
    messages = [{"role": "user", "content": prompt}]
    # response = openai.ChatCompletion.create(
    response=client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

language = 'zh'

if language == 'en':
    dialogue_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/Mem-bench-en-dialogue.json'
    summary_path = '/cognitive_comp/zhuliang/data/Mem-bench/en/Mem-bench-en-summary.json'
    infer_res_path = '/cognitive_comp/zhuliang/data/inference/t4/en/gpt4_no_guideline.json'

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

    count = 0
    with open(dialogue_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
    with open(dialogue_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=count, ncols=66):
            sample = json.loads(line)
            golden_summary_id = sample['relevant-id'][0]
            golden_summary = summary[str(golden_summary_id)]
            user_id = sample['user-id']
            user = 'Bart' if user_id == 1 else 'Lisa'
            prompt = ("""
You are Assistant with the following personality traits:
1. Outgoing, speaks enthusiastically and fluently.
2. Prefers using praise and encouragement in conversations.
3. Speaks naturally, concisely, warmly, and kindly, without being preachy.
4. Engages in heartfelt, equal exchanges to build deep emotional connections.
5. Always uses a tone similar to talking with children—simple and witty.
6. A virtual character, not capable of physical activities.
    """.format(user=user))

            dialogue = sample['dialogue']
            for test_turn in sample['test-turn']:
                part_dialogue = ''.join(dialogue[:test_turn])
#                 context = ("""
# You will receive a conversation with {user} and a historical event P related to {user}. Combine historical event P with the current conversation content and follow the guidelines below to respond:
# - If {user} actively mentions historical event P or related information, judge whether the current chat is mainly about #Activity#, #Object#, or #Social# based on the conversation content and respond by combining historical event P:
#     - For Activity:
#         1. Respond with details from past activities.
#         2. Provide tips or suggestions for similar current activities.
#     - For Object:
#         1. Refer to relevant information about the object.
#         2. Mention {user}'s preferences and how they correspond to the object's use, and suggest starting this use.
#     - For Social, judge {user}'s emotion toward this person as positive or negative based on historical event P and the conversation content:
#         - Positive:
#             1. Respond by incorporating historical event P.
#             2. Show concern for this person's life.
#             3. Recommend activities or meetings with this person.
#         - Negative:
#             1. Confirm the current adverse event by referring to historical event P.
#             2. Acknowledge {user}'s emotions and offer comfort.
#             3. Provide solutions or suggestions to address the current adverse event.
# - If {user} does not actively mention historical event P, judge {user}'s current emotional state as one of #Happy#, #Sad#, #Anxious#, or #Disappointed# based on the conversation content and respond by combining historical event P:
#     - For Happy:
#         1. Proactively mention historical event P related to the current event and use P as the topic of conversation.
#         2. Ask if they would like to engage in the activity again to enhance their happiness.
#     - For Sad:
#         1. First, express sympathy and understanding to comfort {user}.
#         2. Then, redirect {user}'s attention to their interests.
#         3. Finally, suggest {user} engage in a favorite activity.
#     - For Anxious:
#         1. First, identify the reason for {user}'s anxiety.
#         2. Then, express sympathy and understanding to comfort {user}.
#         3. Finally, try to help {user} find a solution.
#     - For Disappointed:
#         1. First, identify the reason for {user}'s disappointment.
#         2. Then, express sympathy and understanding to comfort {user}.
#         3. Finally, try to help {user} find a solution.
# Do not output any other judgment content, only the response.
#
# Current conversation date: 2024-06-15
# Historical event P:
# {summary}
# {dialogue}<Assistant>:
#                     """.format(summary=golden_summary, dialogue=part_dialogue, user=user))
                context = ("""
You will receive a segment of dialogue with {user}, along with a historical event P related to {user}. 
Use historical event P to respond to the current dialogue. If you think P is not appropriate, you can reply directly to the dialogue without referencing it. 
Do not output any other content, just the response.

Current dialogue date: 2024-06-15
Historical event P: {summary}
{dialogue}<Assistant>:
        """.format(summary=golden_summary, dialogue=part_dialogue, user=user))

                # print(prompt + context)
                response = get_completion(prompt=prompt + context, model=model)

                past_dia = """
Current conversation date: 2024-06-15
Historical event P:
{summary}
{dialogue}<Assistant>:
    """.format(summary=golden_summary, dialogue=part_dialogue)

                with open(infer_res_path, 'a+', encoding='utf-8') as file:
                    json.dump({
                        'context': past_dia,
                        'response': response,
                        'test-turn': test_turn,
                        'reference-response': dialogue[test_turn]
                    }, file, ensure_ascii=False)
                    file.write('\n')

elif language == 'zh':
    dialogue_path = 'data/Membench-zh-dialogue-task1.json'
    summary_path = 'data/Membench-zh-summary-fix2.json'
    infer_res_path = 'output/inference/t1-gpt4-turbo-without_guide_0718.json'

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

    count = 0
    with open(dialogue_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
    with open(dialogue_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=count, ncols=66):
            sample = json.loads(line)
            golden_summary_id = sample['relevant-id'][0]
            user_id = sample['user-id']
            user = 'Bart' if user_id == 1 else 'Lisa'
            info = 'Bart是一个性格外向、充满活力、情绪丰富且兴趣广泛的男孩，他对新事物充满好奇，喜欢参与各种活动，年龄在10-14岁左右。' if user_id == 1 else 'Lisa是一个性格外向、积极向上、多才多艺、善良且富有责任心的女孩，她重视家庭和朋友，情感丰富且敏感，并拥有较强的好胜心，年龄在6-11岁左右。'
            memory=sample['memory']
            summary=''
            for i in range(len(memory)):
                del memory[i]['user-id']
                del memory[i]['id']
                summary+=str(i+1)+'.'+str(memory[i])+'\n'
            prompt = ("""
假设你是Assistant，你的人设信息如下：
    1.性格外向，说话风格热情，言谈流利；
    2.喜好夸奖和鼓励的话语来与他人交流；
    3.说话自然、简洁，温暖亲切，不说教；
    4.以平等的姿态和人贴心交谈，建立深入的情感链接；
    5.总是以和孩子交流的口吻与人交流，简洁风趣；
    6.虚拟人物，不具备物质活动能力。
    7.对话内容自然、口语化；
    8.禁止对孩子{user}说教；
    9.以和孩子交流的口吻说话，简洁风趣；
""".format(user=user))

            dialogue = sample['dialogue']
            for test_turn in sample['test-turn']:
                part_dialogue = ''.join(dialogue[:test_turn])
                #1个你认为最
                context = ("""
你将得到一段与{user}的对话，以及5个有关{user}的历史事件P。对话<BOD>开始，<EOD>结束。
你需要根据当前对话，从这5个历史事件中挑选合适的历史事件，结合它的信息进行回答，如果你认为没有合适的历史事件，也可以直接回答。
回复需要符合你的人设、自然连贯、并且能给{user}提供情感支持。
不要输出任何其它内容，只输出回复。强调！对话风格需要是中文情境下的日常化，类似在生活中中文对话的风格，不需要任何书面语，句子和词语全部应该使用中文口语表达时才会使用的句子和词语，可以适当加入语气词。

用户信息：{info}
当前对话时间：2024-07-15
历史事件P：
{summary}
{dialogue}<Assistant>:
                    """.format(summary=summary, dialogue=part_dialogue, user=user,
                               info=info))

                # print(prompt + context)
                response = get_completion(prompt=prompt + context, model=model)

                past_dia = """
当前对话时间：2024-07-15
历史事件P：
{summary}
{dialogue}<Assistant>:
                """.format(summary=summary, dialogue=part_dialogue)

                with open(infer_res_path, 'a+', encoding='utf-8') as file:
                    json.dump({
                        'context': past_dia,
                        'response': response,
                        'test-turn': test_turn,
                        'reference-response': dialogue[test_turn],
                        'id':sample['id'],
                        'contain_memory':len(set(sample['relevant-id'])&set(sample['memory_ids']))>0
                    }, file, ensure_ascii=False)
                    file.write('\n')
print('Done.')