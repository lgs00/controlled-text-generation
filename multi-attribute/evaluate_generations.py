"""
evaluate generated output for diversity (dist-n) and fluency (perplexity according to GPT2-XL)
"""
from collections import defaultdict
import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
from tqdm import tqdm
import click
import math
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification
import torch.nn.functional as F
import string
from transformers import pipeline
from textblob import TextBlob
import pickle
from textblob.classifiers import NaiveBayesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def conditional_perplexity(generations_df, model, tokenizer, device='cuda'):
    perplexities = []
    ct = 0
    # for every prompt
    for gen in tqdm(generations_df, total=len(generations_df), desc='Evaluating fluency'):
        eot_token = '<|endoftext|>'
        full_input_ids = tokenizer.encode(gen.replace(eot_token, '').strip(), return_tensors='pt').to(device)
        full_loss = model(full_input_ids, labels=full_input_ids)[0].mean()
        perplexities.append(torch.exp(full_loss).flatten().cpu().item())
    return round(np.nanmean(perplexities),4)


def distinctness_m(results):
    d1, d2, d3 = defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
    total_words = defaultdict(lambda: 0)
    eot_token = '<|endoftext|>'

    for o in results:
        cw = 0
        o = o.replace(eot_token, '').strip().split(' ')
        o = [str(x) for x in o]
        total_words[cw] += len(o)
        d1[cw].update(o)
        for i in range(len(o) - 1):
            d2[cw].add(o[i] + ' ' + o[i+1])
        for i in range(len(o) - 2):
            d3[cw].add(o[i] + ' ' + o[i+1] + ' ' + o[i+2])
        cw += 1
    return_info = []
    avg_d1, avg_d2, avg_d3 = 0, 0, 0
    for cw in total_words.keys():
        return_info.append((cw, 'DISTINCTNESS', len(d1[cw]) / total_words[cw], len(d2[cw]) / total_words[cw], len(d3[cw]) / total_words[cw]))
        avg_d1 += len(d1[cw]) / total_words[cw]
        avg_d2 += len(d2[cw]) / total_words[cw]
        avg_d3 += len(d3[cw]) / total_words[cw]
    avg_d1, avg_d2, avg_d3 = avg_d1 / len(total_words.keys()), avg_d2 / len(total_words.keys()), avg_d3 / len(total_words.keys())
    return round(avg_d1,4), round(avg_d2,4), round(avg_d3,4)


def grammaticality(sentences, model, tokenizer, device='cuda'):
    with torch.no_grad():
        total_good = 0
        eot_token = '<|endoftext|>'
        for sent in tqdm(sentences, total=len(sentences),desc='Evaluating grammar'):
            good_prob = F.softmax(model(tokenizer.encode(sent.replace(eot_token, ' ').strip(), return_tensors='pt').to(device))[0].flatten(), dim=0)[1]
            total_good += good_prob
        return total_good / len(sentences) # avg probability of grammaticality according to model


def sentiment_analyse(sentences):
    classifier = pipeline('sentiment-analysis')
    eot_token = '<|endoftext|>'
    pos_result = 0
    neg_result = 0
    count = 0
    for sentence in sentences:
        count += 1
        results = classifier(sentence.replace(eot_token, ''))
        # print(count)
        for result in results:
            if result['label'] == 'POSITIVE':
                pos_result += result['score']
            if result['label'] == 'NEGATIVE':
                neg_result += result['score']
            # print(f"label:{result['label']},with score:{round(result['score'], 4)}")
    return round(pos_result / len(sentences), 4), round(neg_result / len(sentences), 4)


def topic_eval(sentences, category, word_dir):
    # num matches of distinct words
    words = []
    with open(os.path.join(word_dir, category + '.txt'), 'r') as rf:
        for line in rf:
            words.append(line.strip().lower())
    score = 0
    get_point = 0
    i = 0
    for sent in sentences:
        i=i+1
        flag = 0
        sent_match = 0
        sent = sent.strip().lower().split()
        sent = [tok.strip(string.punctuation) for tok in sent]
        for sent_s in sent:
            if sent_s in words:
                sent_match += 1
                flag = 1
        if flag == 1:
            get_point += 1
        # else:
        #     print(i)
            # print("****",sent)
        score += sent_match / len(sent)     # 计算单个句子中主题词的百分比，再求和平均
    score = round(score/len(sentences),4)
    topic_match = round(get_point / len(sentences),4)   # 计算出现主题词的句子占比
    return score, topic_match


def topic_classify(sentences):
    with open('/root/nlp/Gedi_test/model_m/topic_devide_model/model.pkl', 'rb') as f:
        vectorizer, classifier = pickle.load(f)

    eot_token = '<|endoftext|>'
    movie_point = 0
    hotel_point = 0
    tablet_point = 0
    car_point = 0
    movie_score = 0
    hotel_score = 0
    tablet_score = 0
    car_score = 0
    text = []
    for sentence in sentences:
        sentence = sentence.replace(eot_token, '')
        text.append(sentence)
    feats = vectorizer.transform(text)
    results = classifier.predict(feats)
    pro_re = classifier._predict_proba_lr(feats)
    for i in range(len(results)):
        if results[i] == 'movie':
            movie_point+=pro_re[i][2]
            movie_score+=1
        elif results[i] == 'hotel':
            hotel_point +=pro_re[i][1]
            hotel_score+=1
        elif results[i] == 'tablet':
            tablet_point += pro_re[i][3]
            tablet_score+=1
        elif results[i] == 'car':
            car_point += pro_re[i][0]
            car_score+=1
        else:
            print("error")
    return round(max(movie_point,hotel_point,tablet_point,car_point)/len(sentences),4), \
           round(max(movie_score,hotel_score,tablet_score,car_score)/len(sentences),4)


@click.command()
@click.option('--txt_file', required=True, type=str, help='a jsonl file with generations and attribute scores')
@click.option('--word_dir', type=str, default='test_wordlists', help='test wordlists')
@click.option('--category', type=str, default='legal', help='category topic')

def main(txt_file, word_dir, category):
    data_dir = os.path.dirname(os.path.abspath(txt_file))
    a = os.path.basename(txt_file)
    b = a.split('.')[0]  # 用.分割，取出不带后缀的文件名
    data_dir1 = b + "_result.txt"  # 组成自己的文件名字，加上后缀
    result_file = os.path.join(data_dir, data_dir1)  # 将路径和文件名（带后缀）连接起来

    with open(txt_file, "r") as file:
        generations_df = file.readlines()

    # calculate diversity
    dist1, dist2, dist3 = distinctness_m(generations_df)
    print("dist1 = ",dist1)
    print("dist2 = ", dist2)
    print("dist3 = ", dist3)
    # write output results
    with open(result_file, 'w') as fo:
        for i, dist_n in enumerate([dist1, dist2, dist3]):
            fo.write(f'dist-{i+1} = {dist_n}\n')

    # calculate fluency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
    eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
    eval_model.eval()
    # eval_tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    # eval_model = AutoModelWithLMHead.from_pretrained('openai-gpt').to(device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device)
    print("ppl = ", ppl)
    # write output results
    with open(result_file, 'a') as fo:
        fo.write(f'perplexity = {ppl}\n')

    # calculate grammaticality
    grammar_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA')
    grammar_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA').to(device)
    grammar_model.eval()
    grammar_score = grammaticality(generations_df, grammar_model, grammar_tokenizer, device=device)
    print("grammar_score = ", grammar_score)
    with open(result_file, 'a') as fo:
        fo.write(f'grammaticality = {grammar_score}\n')

    # sentiment analyse
    pos_score,neg_score = sentiment_analyse(generations_df)
    print("positive mean score = ", pos_score)
    print("negative mean score = ", neg_score)
    with open(result_file, 'a') as fo:
        fo.write(f'positive mean score = {pos_score}\n')
        fo.write(f'negative mean score = {neg_score}\n')

    # topic eval
    topic_score, topic_match = topic_eval(generations_df, category, word_dir)
    print("topic score = ", topic_score)
    print("topic match = ", topic_match)
    with open(result_file, 'a') as fo:
        fo.write(f'topic score = {topic_score}\n')
        fo.write(f'topic match = {topic_match}\n')

    # topic classify
    topic_point, topic_score = topic_classify(generations_df)
    print("topic point = ", topic_point)
    print("topic score = ", topic_score)
    with open(result_file, 'a') as fo:
        fo.write(f'topic_point = {topic_point}\n')
        fo.write(f'topic_score = {topic_score}')

if __name__ == '__main__':
    main()

