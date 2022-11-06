from gensim import models
from gensim.models import word2vec
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from nltk.corpus import brown
#数据预处理
# f1 =open("data/TF_all/true_all.txt","r")
# f2 =open("data/TF_all/true_all_process.txt", 'w')
# lines =f1.readlines()  # 读取全部内容
#
# for line in lines:
#     line = line.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '').replace('（', '').replace('（', '').replace('!', '')
#     line = line.strip().split(" ")
#     line = " ".join(line)
#     f2.writelines(line)
# f1.close()
# f2.close()
#
sentences = word2vec.LineSentence("data/TF_all/true_all_process.txt")

# model = word2vec.Word2Vec(sentences, size=200,window=5,min_count=1,iter=10)
# model.save("data/model/word2vec.model")

model = word2vec.Word2Vec.load("/root/nlp/word_model/word2vec.model")
print(model)
# file_name = "data/vocab.json"
# with open(file_name,'r',encoding='utf-8') as myfile:
#     myjson = json.load(myfile)
words = ["—", '–','…','….', '…"', '―' ]
# for key,value in myjson.items():
#     words.append(key.strip('Ġ'))
# # model = models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
model.build_vocab([words],update=True)
model.train([words],total_examples= model.corpus_count,epochs= model.epochs)
print (model.epochs)
model.save("/root/nlp/word_model/word2vec.model")
print(model)
