"""获取停用词"""

stopwords_path = "./corpus/stopwords.txt"

stopwords = [i.strip() for i in open(stopwords_path).readlines()]