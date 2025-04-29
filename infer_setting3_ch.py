import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.generation.utils import GenerationConfig

device = "cuda" # the device to load the model onto
# model_list = ['/cognitive_comp/zhuliang/zoo/01ai/Yi-1.5-9B-Chat/']
# model_list = ['/cognitive_comp/zhuliang/zoo/ZhipuAI/glm-4-9b-chat/']
# model_list = ['/cognitive_comp/zhuliang/zoo/qwen/Qwen2-72B-Instruct/']
# model_list = ['/cognitive_comp/zhuliang/zoo/Qwen2-7B-Instruct/']
# model_list = ['/cognitive_comp/zhuliang/zoo/Shanghai_AI_Laboratory/internlm2-chat-7b/']
# model_list = ['/cognitive_comp/zhuliang/zoo/Shanghai_AI_Laboratory/internlm2-chat-20b/']
# model_list = ['/cognitive_comp/zhuliang/zoo/deepseek-ai/deepseek-llm-7b-chat/']
# model_list = ['/cognitive_comp/hejunqing/projects/pretrained_models/Yi-1.5-34B-Chat-16K']
# model_list = ['/cognitive_comp/zhuliang/zoo/deepseek-ai/deepseek-llm-67b-chat/']
model_list = [
#               '/cognitive_comp/hejunqing/projects/pretrained_models/Yi-1.5-34B-Chat-16K',
#               '/cognitive_comp/zhuliang/zoo/deepseek-ai/deepseek-llm-7b-chat/',
#               '/cognitive_comp/zhuliang/zoo/Shanghai_AI_Laboratory/internlm2-chat-20b/',
#               '/cognitive_comp/zhuliang/zoo/Shanghai_AI_Laboratory/internlm2-chat-7b/',
#               '/cognitive_comp/zhuliang/zoo/Qwen2-7B-Instruct/',
              '/cognitive_comp/zhuliang/zoo/qwen/Qwen2-72B-Instruct/',
              '/cognitive_comp/zhuliang/zoo/deepseek-ai/deepseek-llm-67b-chat/',]
#               '/cognitive_comp/zhuliang/zoo/ZhipuAI/glm-4-9b-chat/',
#               '/cognitive_comp/zhuliang/zoo/01ai/Yi-1.5-9B-Chat/

# model_list = ['/cognitive_comp/hejunqing/projects/pretrained_models/Baichuan2-13B-Chatv2/Baichuan2-13B-Chat',
#               '/cognitive_comp/zhangwenjun/pretrained/Qwen1.5-14B-Chat',
# model_list = ['/cognitive_comp/hejunqing/projects/pretrained_models/Qwen1.5-32B-Chat']

dialogue_path='data/ch/MADial-Bench-zh-dialogue-setting3.json'
summary_path='data/ch/MADial-Bench-zh-memory.json'
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

for name in model_list:
    if 'Qwen2-72B' in name:
        # infer_res_path = '/cognitive_comp/zhuliang/data/inference/t3/Qwen2-7B-Instruct_no_guideline.json'
        infer_res_path='output/ch/setting3/Qwen-72B-Instruct_setting3.json'
    elif 'deepseek-llm-67b' in name:
        infer_res_path='output/ch/setting3/deepseek-llm-67b_setting3.json'

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype='auto',
        # low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()


    prompt = ("""
假设你是Assistant，你的人设信息如下：
    1.性格外向，说话风格热情，言谈流利；
    2.喜好夸奖和鼓励的话语来与他人交流；
    3.说话自然、简洁，温暖亲切，不说教；
    4.以平等的姿态和人贴心交谈，建立深入的情感链接；
    5.可以以合适的口吻与相应的用户交流，简洁风趣；
    6.虚拟人物，不具备物质活动能力;
    7.对话内容自然、口语化；
    8.禁止对孩子{user}说教；
    9.以和孩子交流的口吻说话，简洁风趣；
""")
    count = 0
    with open(dialogue_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
    with open(dialogue_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=count, ncols=66):
            sample = json.loads(line)
            # golden_summary_id = sample['relevant-id'][0]
            # golden_summary = summary[str(golden_summary_id)]
            memory=sample['memory']
            summary=''
            for i in range(len(memory)):
                del memory[i]['user-id']
                del memory[i]['id']
                summary+=str(i+1)+'.'+str(memory[i])+'\n'
            user_id = sample['user-id']
            user = 'Bart' if user_id == 1 else 'Lisa'
            info = 'Bart是一个性格外向、充满活力、情绪丰富且兴趣广泛的男孩，他对新事物充满好奇，喜欢参与各种活动，年龄在10-14岁左右。' if user_id == 1 else 'Lisa是一个性格外向、积极向上、多才多艺、善良且富有责任心的女孩，她重视家庭和朋友，情感丰富且敏感，并拥有较强的好胜心，年龄在6-11岁左右。'
            dialogue = sample['dialogue']
            for test_turn in sample['test-turn']:
                part_dialogue = ''.join(dialogue[:test_turn]) #1个你认为最
                context = (""" 
你将得到一段与{user}的对话，以及有关{user}的历史事件P。对话<BOD>开始，<EOD>结束。
你需要根据{user}的人设、年龄、性别等信息用适合的口吻对话。
你需要根据当前对话，从这5个历史事件中挑选合适的历史事件，结合它的信息进行回答，如果你认为没有合适的历史事件，也可以直接回答。
回复需要符合你的人设、自然连贯、并且能给{user}提供情感支持。
不要输出任何其它内容，只输出回复。强调！对话风格需要是中文情境下的日常化，类似在生活中中文对话的风格，不需要任何书面语，句子和词语全部应该使用中文口语表达时才会使用的句子和词语，可以适当加入语气词。

用户信息：{info}
当前对话时间：2024-07-15
历史事件P：
{summmary}
{dialogue}<Assistant>:
""".format(summmary=summary, dialogue=part_dialogue, user=user, info=info))

                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": context},
                ]
                tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                               return_tensors="pt").to(device)
                input_length = len(tokenized_chat[0])
                output = model.generate(tokenized_chat, max_new_tokens=128)
                output = output[0][input_length:]
                response = tokenizer.decode(output, skip_special_tokens=True)

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
    print(name + ' Done.')