import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse
import gensim.downloader as api

glove_encoder = api.load("glove-wiki-gigaword-300")

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
