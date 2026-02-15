"""
处理召回的语料
"""
import json
from utils import cut
from tqdm import tqdm

def get_q_info(q):
    cut_by_word = cut(q,by_word=True) #单个字分词
    cut_temp = cut(q,use_seg=True) #
    entity = [i[0] for i in cut_temp if i[-1]=="kc"] #主体
    _cut = [i[0] for i in cut_temp] #分词
    return cut_by_word,_cut,entity

def process_corpsu():
    fq = open("./corpus/recall/Q.txt").readlines()
    fa = open("./corpus/recall/A.txt").readlines()
    qa_dict = {}
    for q,a in tqdm(zip(fq,fa),total=len(fq)):
        q,a = q.strip(),a.strip()
        qa_dict[q] = {}
        cut_by_word, cut, entity = get_q_info(q)
        qa_dict[q]["cut"] = cut
        qa_dict[q]['cut_by_word'] = cut_by_word
        qa_dict[q]["entity"] = entity
        qa_dict[q]["ans"] = a
    # print(qa_dict)
    with open("./corpus/recall/qa_dict.json","w") as f:
        f.write(json.dumps(qa_dict,ensure_ascii=False,indent=2))
        # json.dump(qa_dict,f)
