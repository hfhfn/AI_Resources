"""
执行corpus中数据处理的函数
"""
# from corpus.classify.process_classify_corpus import start_process,data_split
from corpus.chatbot.process_chatbot_corpus import process_xiaohuangji,start_process
from corpus.recall.process_recall_corpus import process_corpsu

if __name__ == '__main__':
    # start_process()
    # data_split()
    # process_xiaohuangji()
    # start_process()
    # interface()
    process_corpsu()