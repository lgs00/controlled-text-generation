from fast_bleu import BLEU, SelfBLEU
import numpy as np
import nltk
import numpy as np
import csv
import random
import pickle
from nltk import ngrams
from collections import Counter
import argparse
from ngram_utils import ngrams
# nltk.download('punkt')
from textblob.classifiers import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC,SVC
# from kashgari.embeddings import BERTEmbedding
# from kashgari.tasks.classification.models import CNN_Model
from xgboost import XGBClassifier
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def preprocess_text(file):
        list_of_gen = []
        for line in file.readlines():
                words = nltk.word_tokenize(line)        # 对每个句子进行分词
                for word in words:
                        list_of_gen.append(word)
                file.close()

        size = len(list_of_gen)
        idx_list = [idx + 1 for idx, val in enumerate(list_of_gen) if val == '|endoftext|']

        res = [list_of_gen[i: j] for i, j in
                zip([0] + idx_list, idx_list +
                ([size] if idx_list[-1] != size else []))] [:-1]

        gen = []
        for sent in res:
                sent = remove_values_from_list(sent, '<')
                sent = remove_values_from_list(sent, '|endoftext|')
                sent = remove_values_from_list(sent, '>')
                gen.append(sent)
        return gen


def proc_and_binarize(dir):
    fid = open(dir + "/train.tsv")
    train = fid.read()
    train = train.split("\n")[:-1]

    fid = open(dir + "/test.tsv")
    test = fid.read()
    test = test.split("\n")[:-1]  # 对测试集进行划分

    train_text = []
    train_label = []

    test_text = []
    test_label = []

    for i in range(0, len(train)):
        line = train[i].split("\t")  # 对测试集再划分为标签和数据
        label = line[0]  # 取出标签
        if not (len(line) == 2):
            print("skipping " + str(i))
            continue
        text = line[1]
        if text[0] == "\"":
            text = text[1:-1]
        if text[0] == " ":
            text = text[1:]
        train_text.append(text)
        train_label.append(label)

    for i in range(0, len(test)):
        line = test[i].split("\t")
        if not (len(line) == 2):
            print("skipping " + str(i))
            continue
        label = line[0]
        text = line[1]
        if text[0] == "\"":
            text = text[1:-1]
        if text[0] == " ":
            text = text[1:]
        test_text.append(text)
        test_label.append(label)

    return train_text, train_label, test_text, test_label


def topic_classify(sentences):
    with open('/root/nlp/Gedi_test/model_m/topic_devide_model/model.pkl', 'rb') as f:
        vectorizer, classifier = pickle.load(f)

    eot_token = '<|endoftext|>'
    movie_point = 0
    hotel_point = 0
    tablet_point = 0
    car_point = 0
    count = 0
    text = []
    for sentence in sentences:
        sentence = sentence.replace(eot_token, '')
        text.append(sentence)
    feats = vectorizer.transform(text)
    results = classifier.predict(feats)
    for i in range(len(results)):
        if results[i] == 'movie':
            movie_point+=1
        elif results[i] == 'hotel':
            hotel_point +=1
        elif results[i] == 'tablet':
            tablet_point += 1
        elif results[i] == 'car':
            car_point += 1
        else:
            print("error")
    return round(movie_point / len(sentences), 4), round(hotel_point / len(sentences), 4), \
           round(tablet_point / len(sentences), 4), round(car_point / len(sentences), 4)


if __name__ == '__main__':
    dir = "/root/nlp/Gedi_test/data/topic_devide"

    # csv.writer(open("/home/liuguisheng/GeDi/data/AG-news/train1.tsv", 'w+'), delimiter='\t').writerows(csv.reader(open("/home/liuguisheng/GeDi/data/AG-news/train.csv")))
    # csv.writer(open("/home/liuguisheng/GeDi/data/AG-news/test1.tsv", 'w+'), delimiter='\t').writerows(csv.reader(open("/home/liuguisheng/GeDi/data/AG-news/test.csv")))

    train_text, train_label, test_text, test_label = proc_and_binarize(dir)
    # embedding = BERTEmbedding('/root/nlp/word_model/cased_L-12_H-768_A-12/', task=kashgari.CLASSIFICATION)
    # model = CNN_Model(embedding)
    # train_samples = list(zip(train_text, train_label))
    # model.fit(test_text, train_label)

    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(train_text)
    # train_feats = vectorizer.transform(train_text)
    # classifier = XGBClassifier()
    # classifier.fit(train_feats, train_label)
    # test_feats = vectorizer.transform(test_text)
    # acc = classifier.score(test_feats, test_label)
    # print(acc)
    # classifier = NaiveBayesClassifier(train_samples)
    # test_samples = list(zip(test_text, test_label))
    # model.evaluate(test_x, test_label)
    # acc = classifier.accuracy(test_samples)
    # with open('/root/nlp/Gedi_test/model_m/topic_devide_model/model0.pkl', 'wb') as f:
    #     data = [vectorizer, classifier]
    #     pickle.dump(data, f)
    # vectorizer, classifier = None, None
    # txt_file = "/root/nlp/Gedi_test/generations/0402/5_ad2_movie_sentiment/movie_negative.txt"
    # with open(txt_file, "r") as file:
    #     generations_df = file.readlines()
    with open('/root/nlp/Gedi_test/model_m/topic_devide_model/model.pkl', 'rb') as f:
        vectorizer, classifier = pickle.load(f)
    texts = ["Dr. Krellman wants to save his son Julio who\'s dying of heart disease. He decides a heart transplant with an ape will cure his son ( no--I\'m not kidding ) . He does the transplant and ( somehow ) his son changes from a frail guy into a muscle-bound man with a dime store mask that ( sort of ) resembles an ape! Naturally he gets out, kills men, tears the clothes of women and wreaks havoc. This is all inter cut with the boring romance of police lt. Arturo Martinez and lady wrestler Lucy Ossorio. We also get pointless female wrestling sequences that add nothing to the plot. It all ends by copying the end of ""King Kong""! This is ( obviously ) a pretty stupid movie. The plot makes little sense, there\'s the gratuitous female nudity ( a staple of any exploitation film ) and VERY graphic gore that looks laughably fake ( except for the open heart transplant ) . Still this does have merit. The whole cast takes everything dead serious and actually aren\'t too bad as actors. Also the dubbed in dialogue was ( for a film like this ) well done and interesting with surprisingly good dubbing. Also I saw an excellent DVD print with bright strong color ( which helps ) . We\'re not talking a classic here but an OK exploitation movie.",'room clean','Cuase i work on cars everyday, this fucking 5 cylinder turbo engien has more? potential then a ferrari has!','Take? my money Apple!!', ]
    feats = vectorizer.transform(texts)
    results = classifier.predict(feats)
    pro_re = classifier._predict_proba_lr(feats)
    print(pro_re)
    print (results)
    # topic classify
    # topic_movie, topic_hotel, topic_tablet, topic_car = topic_classify(generations_df)
    # print("topic movie = ", topic_movie)
    # print("topic hotel = ", topic_hotel)
    # print("topic tablet = ", topic_tablet)
    # print("topic car = ", topic_car)
    # print(acc)




