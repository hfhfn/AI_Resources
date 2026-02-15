"""
处理和准备意图识别需要的语料
分词后空格链接\t__label__QA/chat
"""
from utils import cut
from tqdm import tqdm
import json
import random


def save_line(line,label,f_by_word,f):

    #单个字分词后带保存的line
    line_word = " ".join(cut(line,by_word=True))+"\t__label__"+label+"\n"
    f_by_word.write(line_word)

    #写入data.txt中
    line = " ".join(cut(line,by_word=False))+"\t__label__"+label+"\n"
    f.write(line)



def process_xiaohuangji(f_by_word,f):
    data_path = "./corpus/classify/小黄鸡未分词.conv"

    groups = []  #[[q,a],[q,a],[q,a]]
    group = []
    bar = tqdm(open(data_path).readlines(),desc="小黄鸡数据读取...")
    for line in bar:
        if line.startswith("E"):
            if group:
                groups.append(group)
                group = []
        elif line.startswith("M"):
            group.append(line)
    if group:
        groups.append(group)

    for group in tqdm(groups,desc="小黄鸡数据保存..."):  #一个group就是一个问答对
        if len(group) == 2:
            q = group[0][1:].strip().lower()
            # a = group[1].strip().lower()
            if len(q)>0:
                save_line(q,"chat",f_by_word,f)


def process_qa(f_by_word,f):
    data_path1  = "./corpus/classify/爬虫抓取的问题.csv"
    data_path2 = "./corpus/classify/手动构造的问题.json"
    data_path3 = "./corpus/classify/merged_q.txt"

    data_set = set()
    for line in tqdm(open(data_path1).readlines(),desc="data_path1数据读取"):
        data_set.add(line.strip().lower())
    for line in tqdm(open(data_path3).readlines(),desc="data_path3数据读取..."):
        data_set.add(line.strip().lower())
    for key,values in  tqdm(json.load(open(data_path2,"r")).items(),desc="data_path2数据读取..."):
        for v in values:
            for line in v:
                data_set.add(line.strip().lower())
    for line in tqdm(data_set,desc="qa数据保存..."):
        save_line(line,"qa",f_by_word,f)


def start_process():
    f_by_word = open("./corpus/classify/data_by_word.txt", "a")
    f = open("./corpus/classify/data.txt", "a")
    process_xiaohuangji(f_by_word,f)
    process_qa(f_by_word,f)

    f_by_word.close()
    f.close()


def data_split():
    train_path = open("./corpus/classify/data_by_word_train.txt","a")
    test_path = open("./corpus/classify/data_by_word_test.txt","a")
    for line in tqdm(open("./corpus/classify/data_by_word.txt", "r").readlines()):
        if random.random()>0.8:
            test_path.write(line)
        else:
            train_path.write(line)

    train_path = open("./corpus/classify/data_train.txt", "a")
    test_path = open("./corpus/classify/data_test.txt", "a")
    for line in tqdm(open("./corpus/classify/data.txt", "r").readlines()):
        if random.random() > 0.8:
            test_path.write(line)
        else:
            train_path.write(line)


