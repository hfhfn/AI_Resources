"""
构造召回的模型
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci
from utils import cut
import json

def prepar_recall_datas():
    qa_dict = json.load(open("./corpus/recall/qa_dict.json"))
    q_list = []
    q_cut = []
    for i in qa_dict:
        q_list.append(i)
        q_cut.append(" ".join(qa_dict[i]["cut"])) #分词之后的问题 [sentence,sentence,....]

    tfidf_vec = TfidfVectorizer()
    q_vector = tfidf_vec.fit_transform(q_cut) #得到问题的向量

    #准备搜索的索引
    cp = ci.MultiClusterIndex(q_vector,q_list)

    return tfidf_vec,cp,qa_dict


def get_search_result(input):
    tfidf_vec, cp, qa_dict = prepar_recall_datas()
    entity = []
    input_cut = []
    for word,seg in cut(input,by_word=False,use_seg=True):
        input_cut.append(word)
        if seg == "kc":
            entity.append(word)
    # 1. 得到用户问题的向量
    input_vector = tfidf_vec.transform([" ".join(input_cut)])
    #  2. 计算相似度
    result = cp.search(input_vector,k=10,k_clusters=10,return_distance=True)
    print(result)


