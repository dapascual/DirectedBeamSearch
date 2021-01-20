import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse
import gensim.downloader as api

encode_50keywords = True
encode_articles = False

glove_encoder = api.load("glove-wiki-gigaword-300")

if encode_50keywords == True:
    
    file1 = open(str(os.path.dirname(os.path.abspath(__file__))) +
                 "/data/50_keywordsets_eval/word_sets.txt", "r+")

    lines = file1.readlines()

    i=0
    for line in lines:
        keywords = list(line.strip().split(", "))
        glove_words = []
        for word in keywords:
            print(word)
            glove = glove_encoder[word]
            glove_words.append(glove)
        save_path = str(os.path.dirname(
            os.path.abspath(__file__))) + '/data/50_keywordsets_eval/set_' +str(i) + '.npy'
        np.save(save_path, glove_words)
        i=i+1

if encode_articles == True:    

    for n in [4, 5, 8, 9, 10, 12, 13, 14, 15, 16]:
        print(n)
        file1 = open(str(os.path.dirname(os.path.abspath(__file__))) +
                     "/data/keyword_to_articles/test_" + str(n) + ".txt", "r+")

        lines = file1.readlines()

        keywords = list(lines[2].strip().split(", "))
        print(keywords)
        glove_words = []
        for word in keywords:
            glove = glove_encoder[word]
            glove_words.append(glove)

        save_path = str(os.path.dirname(
            os.path.abspath(__file__))) + '/data/keyword_to_articles/test_' +str(n) + '.npy'
        np.save(save_path, glove_words)
    

