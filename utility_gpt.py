import numpy as np
import os
import pandas as pd
import scipy.io as sio
import math
import torch
import spacy
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from nltk.stem import PorterStemmer
porter = PorterStemmer()

def glove_encode(glove_encoder, word):
    return glove_encoder(word)



def checker(string):
    string = string.replace("'ve", '')
    string = string.replace("@", '')
    string = string.replace("'re", '')
    string = string.replace("'d", '')
    string = string.replace("?", '')
    string = string.replace("'s", '')
    string = string.replace(":", '')
    string = string.replace("!", '')
    string = string.replace('"', '')
    string = string.replace(".", '')
    string = string.replace("--", '')
    string = string.replace("'", '')
    string = string.replace(",", '')
    string = string.replace(';', '')
    string = string.replace('‘', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace('\'', '')
    string = string.replace(' ', '')
    return(string)


## Pytorch
def converter_table_glove():
    import gensim.downloader as api
    glove_encoder = api.load("glove-wiki-gigaword-300")

    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/data/converter_table_glove'

    # load gpt-2 model
    #model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #model.eval()

    holder = np.zeros((50257, 300))

    # translate every word from the gpt-2 space into a glove representation
    for i in range(50257):
        try:
            word = tokenizer.decode([i])
            word = checker(word.strip().lower())
            glove = glove_encoder[word]
            holder[i, :] = glove
        except:
            word = tokenizer.decode([i])
            holder[i, :] = np.zeros((300)) #+ 500

    # Save all 50'000 glove representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Table was generated')
    


def count_word_stem_one(word, sequence):
    #print ("Sequence", sequence)
    sequence = sequence.split()

    word_count = 0
    word_stem = porter.stem(word)
    for s_word in sequence:
        s_word_stem = porter.stem(s_word)
        if(s_word_stem == word_stem):
            word_count = 1
            break

    return word_count

def count_word_stem(word, sequence):
    #print ("Sequence", sequence)
    sequence = sequence.split()
    word_count = 0

    word_stem = porter.stem(word)

    for s_word in sequence:
        s_word_stem = porter.stem(s_word)
        #print(s_word_stem)
        if(s_word_stem == word_stem):
            word_count += 1

    return word_count

# A score function for the quality of the sentence
def evaluate_quality(sequence, word, related_count, perplexity, guide, temp):
    # we aim for one ocurance of the word,  and low perplexity
    w_1 = 1
    w_3 = 0.001 
    c_star = 2

    if(word == ""):
        quality_score = math.exp(-(w_1*(c_star) + w_3*perplexity))
        return quality_score

    quality_score = 0
    word_count = count_word_stem(word, sequence)
    

    if(word_count != 0) and guide:
        quality_score = math.exp(-(w_1*word_count + w_3*perplexity))
    else:
        quality_score = math.exp(-(w_1*(c_star) + w_3*perplexity))

    quality_score = quality_score/temp
    # DEBUG
    #print("txt, quality_score, word_count, rel_count, ppl", sequence, quality_score, word_count, related_count, perplexity)

    return quality_score, word_count

