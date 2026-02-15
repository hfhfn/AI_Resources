"""
测试classify下的所有的api
"""
from classify.build_model import prepar_model,test_model
from classify.classify import Classify
if __name__ == '__main__':
    prepar_model()
    test_model()
    # classify = Classify()
    # classify.predict("python 是 什么")