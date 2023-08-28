#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import copy
import json
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import collections
from convert_raw_data import *

# In[4]:


def convert_raw_data(data_dir, save_data=False, save_dict=False,use_clean=False):
    """
    1、5 折交叉验证构造带标签数据的 distant trigger （长度大于一，并且出现次数大于一）
    2、将复赛数据 8.5 : 1.5 划分训练 / 验证数据；
    3、构建先验知识词典 ---- (tense_prob, polarity_prob)；
    4、构造 trigger 词典 （所有长度等于2的 trigger，去重后构造一个 trigger dict）
    """
    random.seed(123)

    raw_dir = os.path.join(data_dir, 'raw_data')#路径拼接 os.path.join
    mid_dir = os.path.join(data_dir, 'mid_data')
    if not os.path.exists(mid_dir):
        os.mkdir(mid_dir)#路径创造（单级目录）
#     print(raw_dir)../../data/final/raw_data
#     print(mid_dir)../../data/final/mid_data
#     print(data_dir)../../data/final
    


    test_examples = load_examples(os.path.join(raw_dir, 'sentences.json'))
#     print(test_examples[0])
    if not use_clean:
        stack_examples = load_examples(os.path.join(raw_dir, 'raw_stack.json'))
        preliminary_examples = load_examples(os.path.join(raw_dir, 'raw_preliminary.json'))
    else:
        stack_examples = load_examples(os.path.join(raw_dir, 'raw_stack_clean.json'))
        preliminary_examples = load_examples(os.path.join(raw_dir, 'raw_preliminary_clean.json')) 

    kf = KFold(10)

    triggers = {}

    nums = 0

    # 利用raw_stack.json文件内容5折交叉构造 distant trigger
    
    for _now_id, _candidate_id in kf.split(stack_examples):
        now = [stack_examples[_id] for _id in _now_id]
        candidate = [stack_examples[_id] for _id in _candidate_id]
#         print(candidate)
#         print(now)

        now_triggers = {}

        for _ex in now:#在now里
            for _event in _ex['events']:#把event拿出来
                tmp_trigger = _event['trigger']['text']#把event里的，trigger中的text拿出来，是‘发布’
                # distant trigger 选取长度为2的
                if len(tmp_trigger) != 2:
                    continue
                if tmp_trigger in now_triggers:#如果‘发布’在now_triggers字典里，发布加1
                    now_triggers[tmp_trigger] += 1
                else:
                    now_triggers[tmp_trigger] = 1

        for _ex in candidate:#在候选里
            tmp_sent = _ex['sentence']#把sentence出来
            candidate_triggers = []
            for _t in now_triggers.keys():#now_triggers字典里取键
                if _t in tmp_sent :#如果键在sentence里，加在candidate_triggers列表中
                    #and now_triggers[_t] > 1
                    candidate_triggers.append(_t)

            for _event in _ex['events']:
                tmp_trigger = _event['trigger']['text']
                # distant trigger 选取长度为2的
                if len(tmp_trigger) != 2:
                    continue
                if tmp_trigger in triggers:
                    triggers[tmp_trigger] += 1
                else:
                    triggers[tmp_trigger] = 1

            _ex['distant_triggers'] = candidate_triggers#生成distant_triggers项

            if len(candidate_triggers) > nums:
                nums = len(candidate_triggers)

    print(nums)
    nums = 0
    
    #在raw_preliminary.json文件中
    for _ex in preliminary_examples:
        tmp_sent = _ex['sentence']
        candidate_triggers = []
        for _t in triggers.keys():
            if _t in tmp_sent :
                #and triggers[_t] > 1
                candidate_triggers.append(_t)

        _ex['distant_triggers'] = candidate_triggers
        if len(candidate_triggers) > nums:
            nums = len(candidate_triggers)

    print(nums)
    nums = 0

    # 利用sentences.json内容构造 test 的 distant trigger
    for _ex in test_examples:
        tmp_sent = _ex['sentence']
        candidate_triggers = []
        for _t in triggers.keys():
            if _t in tmp_sent:
#             if _t in tmp_sent and triggers[_t] > 1:需要在训练集里出现两次及以上才被抽取为distanttrigger，更改一下！！！！
                candidate_triggers.append(_t)

        _ex['distant_triggers'] = candidate_triggers
        if len(candidate_triggers) > nums:
            nums = len(candidate_triggers)

    triggers = dict(sorted(triggers.items(), key=lambda x: x[1], reverse=True))
#     print(triggers)

    tense = {}
    polarity = {}
    counts = 0.

    print(nums)

    for _ex in tqdm(stack_examples, desc='raw data convert'):#raw_stack.json + distant_triggers
        
        _ex.pop('words')
        
        for _event in _ex['events']:
            tmp_tense = _event['tense']
            tmp_polarity = _event['polarity']
            counts += 1

            if tmp_tense not in tense:
                tense[tmp_tense] = 1
            else:
                tense[tmp_tense] += 1

            if tmp_polarity not in polarity:
                polarity[tmp_polarity] = 1
            else:
                polarity[tmp_polarity] += 1

    def build_map(info):
        info = {key: info[key] / counts for key in info.keys()}
        info2id = {'map': {}, 'prob': []}
        
        for idx, key in enumerate(info.keys()):
            info2id['map'][key] = idx
            info2id['prob'].append(info[key])
        return info2id

    tense2id = build_map(tense)
    polarity2id = build_map(polarity)
    
    triggers_dict = {key: idx + 1 for idx, key in enumerate(triggers.keys())}

    train, dev = train_test_split(stack_examples, shuffle=True, random_state=123, test_size=0.15)
    #test_size：如果是浮点数，在0-1之间，表示样本占比,random_state：是随机数的种子
    #当shuffle=True且randomstate 取整数，划分得到的是乱序的子集，且多次运行语句（保持randomstate值不变），得到的四个子集不变
    print(len(train), len(dev))#2087 369


    if save_data:
        save_info(raw_dir, stack_examples, 'stack')#raw_stack.json
        save_info(raw_dir, train, 'train')
        save_info(raw_dir, dev, 'dev')
        save_info(raw_dir, test_examples, 'test')#sentences.json
        save_info(raw_dir, preliminary_examples, 'preliminary_stack')#raw_preliminary.json

    if save_dict:
        save_info(mid_dir, tense2id, 'tense2id')
        save_info(mid_dir, polarity2id, 'polarity2id')
        save_info(mid_dir, triggers_dict, 'triggers_dict')
if __name__ == '__main__':
    stack = []
    with open(os.path.join("../../data/preliminary/raw_data/", 'stack.json'), encoding='utf-8') as f:
        for line in f.readlines():#读取所有行
            stack.append(json.loads(line.strip()))
    
    new_stack=[]
    for sample in tqdm(stack):
        new_sample={'sentence':sample['text'],'events':[],'distant_triggers':sample['distant_trigger']}
        for event in sample['labels']:
            new_event={'trigger':{'text':event['trigger'][0],'length':len(event['trigger'][0]),'offset':event['trigger'][1]},                       'tense': '过去','polarity': '肯定',                       'arguments':[]}
            for role in event.keys():
                if role=='trigger':
                    continue
                if isinstance(event[role],list):
                    role_name=role if role!='location' else 'loc'
                    if role_name=='subject':
                        role_name='object'
                    elif role_name=='object':
                        role_name='subject'
                    new_event['arguments'].append({'role':role_name, 'text': event[role][0], 'offset': event[role][1], 'length':len(event[role][0])})
            new_sample['events'].append(new_event)
        new_stack.append(new_sample)
    
    #生成raw_preliminary.json
    save_info("../../data/final/raw_data/",new_stack,'raw_preliminary')#把preliminary/stack.json格式转换成raw_preliminary.json格式
    raw_dir="../../data/final/raw_data/"
    
    raw_pre_examples = load_examples(os.path.join(raw_dir, 'raw_preliminary.json'))
    #生成复赛样式的初赛数据
    raw_pre_examples,nums=clean_data(raw_pre_examples)#清洗raw_preliminary.json数据，生成raw_preliminary_clean.json
    print(nums)
    #对初赛数据进行清洗
    raw_examples = load_examples(os.path.join(raw_dir, 'raw_stack.json'))
    raw_examples,nums=clean_data(raw_examples)#清洗raw_stack.json数据，生成raw_stack_clean.json
    print(nums)
    
    save_info("../../data/final/raw_data/",raw_pre_examples,'raw_preliminary_clean')
    save_info("../../data/final/raw_data/",raw_examples,'raw_stack_clean')

    convert_raw_data('../../data/final', save_data=True, save_dict=True,use_clean=True)
    #初赛数据的样本从复赛数据中读取distant triggers
    #输入数据为preliminary/stack.json，和final/raw_stack.json
