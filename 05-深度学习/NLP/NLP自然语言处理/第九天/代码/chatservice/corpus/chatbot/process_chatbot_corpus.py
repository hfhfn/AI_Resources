"""
处理闲聊机器人的语料
"""
import re
from utils import cut
from tqdm import tqdm

def clean_line(line):
    """处理句子中的标点符号"""
    line = re.sub("\^.*?\^","\^***\^",line)
    line = re.sub("\(.*?\)","\(***\)",line)

    result = []  #【【】，【】，[word,True,False],[word,True]】
    temp =""
    for word in line:
        if word.isalpha() or word.isdigit():
            if len(temp)>0:
                result.append([temp,True])
                temp = "" #如果temp里面只有一个字符
            result.append([word,False])
        else:
            temp += word

    if len(temp) > 0:
        result.append([temp, True])

    #把result中第二个位置为True的进行替换
    if result[0][-1]:
        result = result[1:]
    #经过上一步后，有可能为空列表
    if len(result)>0:
        if result[-1][-1]:
            result = result[:-1]+[["。",False]]

    final_result = []
    for i in result:
        if i[-1]: #为标点的情况
            if "!" in i[0] or "！" in i[0]:
                final_result.append(["！",False])
            elif "…" in i[0]:
                final_result.append(["…", False])
            else:
                final_result.append(["，",False])

        else:
            final_result.append(i)
    return "".join([i[0] for i in final_result])


def clean_group(group):
    """
    清理group中的输出
    :param group: [q,a]
    :return: [q,a]/bool
    """
    #判断句子是否为纯标点英文数字,或者是其他的语言--》判断一句话中是否有中文
    if not re.findall("[\u4e00-\u9fa5]",group[0]):
        return False
    if not re.findall("[\u4e00-\u9fa5]",group[1]):
        return False

    #问题中包含`笑话`两个字的
    if re.findall("笑话|糗百|运势|运程",group[0]):
        return False

    #处理连续的多个标点
    group[0] = clean_line(group[0])
    group[1] = clean_line(group[1])

    #小黄鸡，小通
    group[0] = re.sub("小通|鸡鸡","小智",group[0]).strip()
    group[1] = re.sub("小通|鸡鸡","小智",group[1]).strip()

    #判断句子是否为空
    if len(group[0])<1 or len(group[1])<1:
        return False
    return group

def save_group(group,fq,fa,by_word):
    """保存问答对"""

    fq.write(" ".join(cut(group[0],by_word=by_word))+"\n")
    fa.write(" ".join(cut(group[1],by_word=by_word))+"\n")


def process_xiaohuangji(by_word,fq,fa):
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
            group.append(line[1:].strip())
    if group:
        groups.append(group)

    for group in tqdm(groups,desc="小黄鸡数据保存..."):  #一个group就是一个问答对
        group = clean_group(group)
        if not group:
            continue
        # print("q:",group[0])
        # print("a:",group[1])
        # print("*"*30)
        save_group(group,fq,fa,by_word)

def start_process(by_word=True):
    fq = open("./corpus/chatbot/input.txt","a")
    fa = open("./corpus/chatbot/target.txt","a")
    process_xiaohuangji(by_word,fq,fa)


