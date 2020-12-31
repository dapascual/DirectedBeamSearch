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


def load_words():
    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/data/subjects/P01/examples_180concepts_sentences.mat'
    mat_file = sio.loadmat(path)
    words = mat_file['keyConcept']
    return words


def checker(string):
    string = string.replace("'ve", '')
    string = string.replace("@", '')
    string = string.replace("'re", '')
    string = string.replace("malfoy'll", 'malfoy')
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


def files_exist():
    booler = os.path.exists(str(os.path.dirname(os.path.abspath(
        __file__))) + '/look_ups_gpt-2/converter_table.npy')
    booler = booler and os.path.exists(str(os.path.dirname(os.path.abspath(
        __file__))) + '/look_ups_gpt-2/converter_table_glove.npy')
    booler = booler and os.path.isfile(str(os.path.dirname(
        os.path.abspath(__file__))) + '/look_ups_gpt-2/top_ten_embeddings')
    booler = booler and os.path.isfile(str(os.path.dirname(
        os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets_num.npy')
    booler = booler and os.path.isfile(str(os.path.dirname(
        os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets.txt')
    return booler


def generate_look_ups(glove_encoder):
    if not os.path.exists(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/converter_table.npy'):
        converter_table()
    if not os.path.exists(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/converter_table_glove.npy'):
        converter_table_glove(glove_encoder)
    if not os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/top_ten_embeddings'):
        top_ten_embeddings()
    if not os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets.txt'):
        print('First create a text file with top 5 words you want the generation to anchored with.')
    else:
        if not os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets_num.npy'):
            word_sets()


def word_sets():

    lister = []
    words = load_words()
    for i in range(180):
        lister.append(words[i][0][0])

    file1 = open(str(os.path.dirname(os.path.abspath(__file__))) +
                 '/look_ups_gpt-2/word_sets.txt', "r+")
    length = len(file1.readlines())
    file1 = open(str(os.path.dirname(os.path.abspath(__file__))) +
                 '/look_ups_gpt-2/word_sets.txt', "r+")
    holder = np.zeros((length, 5), dtype=int)
    for i in range(length):
        line = file1.readline()
        lines = line.split(',')
        for j in range(5):
            if j == 0:
                numb = lister.index(lines[j].strip())
            else:
                numb = lister.index(lines[j].strip())
            holder[i, j] = numb
    np.save(file=str(os.path.dirname(os.path.abspath(__file__))) +
            '/look_ups_gpt-2/word_sets_num.npy', arr=holder)


# Create the misterious converter table...
def converter_table():
    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/look_ups_gpt-2/converter_table'

    # Load glove embedings dictionary
    embeddings_dict = {}
    # Need glove embeddings dataset!
    # Need glove embeddings dataset!
    with open(str(os.path.dirname(os.path.abspath(__file__))) + "/data/glove_data/glove.42B.300d.txt", 'r') as f:
        for line in f:
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
            except ValueError:
                print(line.split()[0])

    # load gpt-2 model
    model_name = '774M'
    models_dir = str(os.path.dirname(
        os.path.abspath(__file__))) + '/models_gpt'
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)

    holder = np.zeros((50257, 300))

    # translate every word from the gpt-2 space into a glove representation
    for i in range(50257):
        try:
            word = enc.decode([i])
            word = checker(word.strip().lower())
            glove = embeddings_dict[word]
            holder[i, :] = glove
        except:
            word = enc.decode([i])
            holder[i, :] = np.zeros((300)) + 500

    # Save all 50'000 glove representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Converter table was generated')



## Tensorflow...
"""
def converter_table_glove(glove_encoder):
    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/look_ups_gpt-2/converter_table_glove'

    # load gpt-2 model
    model_name = '774M'
    models_dir = str(os.path.dirname(
        os.path.abspath(__file__))) + '/../GPT2/gpt-2/models'
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)

    holder = np.zeros((50257, 300))

    # translate every word from the gpt-2 space into a glove representation
    for i in range(50257):
        try:
            word = enc.decode([i])
            word = checker(word.strip().lower())
            glove = glove_encoder[word]
            holder[i, :] = glove
        except:
            word = enc.decode([i])
            holder[i, :] = np.zeros((300)) + 500

    # Save all 50'000 glove representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Converter table was generated')
"""
## Pytorch
def converter_table_glove(glove_encoder):
    import gensim.downloader as api
    glove_encoder = api.load("glove-wiki-gigaword-300")

    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/look_ups_gpt-2/converter_table_glove'

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
            holder[i, :] = np.zeros((300)) + 500

    # Save all 50'000 glove representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Converter table was generated')
    
def glove_words_table(file_path, glove_encoder):
    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/look_ups_gpt-2/glove_words_table/'

    file1 = open(file_path)
	
    lines = file1.readlines()
    all_glove_words = []
    i = 0
    for line in lines:
        words = list(line.strip().split(", "))
        glove_words = [word_vectors[w] for w in word] #word_vectors[five_words[0]]
        np.save(file=path+"set_"+str(i), arr=glove_words)	
        i += 1   
    
    print('Glove words tables generated')


def find_closest_embeddings_cosine_prox(embedding, embeddings_dict):
    return sorted(embeddings_dict.keys(), key=lambda word: cosine(embeddings_dict[word], embedding))


def top_ten_embeddings():

    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/look_ups_gpt-2/top_ten_embeddings_two'

    word = load_words()
    words = load_words_and_glove()

    embeddings_dict = {}
    # Need glove embeddings dataset!
    with open(str(os.path.dirname(os.path.abspath(__file__))) + "/glove.42B.300d.txt", 'r', encoding='utf8') as f:
        for line in f:
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
            except ValueError:
                print(line.split()[0])
    embbeddings = []

    for i in range(words.shape[0]):
        top_ten = find_closest_embeddings_cosine_prox(
            words[i, :], embeddings_dict)[:11]
        embbeddings.append(top_ten)
    df = pd.DataFrame(embbeddings)
    df.to_csv(path, index=False)


def isSubset(subarraies, array):
    counter = 0
    for subarray in subarraies:
        for i in range(len(array)):
            for j in range(len(subarray)):
                if i+j < len(array):
                    if array[i+j] == subarray[j]:
                        if j == len(subarray)-1:
                            counter += 1
                    else:
                        break
    return counter


def tokens_from_words(Numbers, enc):
    #model_name = '774M'
    #models_dir = str(os.path.dirname(
    #    os.path.abspath(__file__))) + '/../GPT2/gpt-2/models'
    #models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    #enc = encoder.get_encoder(model_name, models_dir)
    word = load_words()
    container = []
    for Nr in Numbers:
        container.append(enc.encode(' ' + word[Nr][0][0]))
    return container


def load_words_and_glove():
    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/data/glove_data/180_concepts.mat'
    mat_file = sio.loadmat(path)
    glove = mat_file['data']
    return glove

def load_words_and_glove_new():

    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/look_ups_gpt-2/top_ten_embeddings_two'

    word = load_words()
    words = load_words_and_glove()

    embeddings_dict = {}
    # Need glove embeddings dataset!
    with open(str(os.path.dirname(os.path.abspath(__file__))) + "/glove.42B.300d.txt", 'r', encoding='utf8') as f:
        for line in f:
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
            except ValueError:
                print(line.split()[0])
    embbeddings = []

    for i in range(words.shape[0]):
        top_ten = find_closest_embeddings_cosine_prox(
            words[i, :], embeddings_dict)[:11]
        embbeddings.append(top_ten)
    df = pd.DataFrame(embbeddings)
    df.to_csv(path, index=False)


def related_words(enc):
    top_ten = pd.read_csv(str(os.path.dirname(os.path.abspath(
        __file__))) + '/look_ups_gpt-2/top_ten_embeddings')
    #model_name = '774M'
    #models_dir = str(os.path.dirname(
    #    os.path.abspath(__file__))) + '/../GPT2/gpt-2/models'
    #models_dir = os.path.expanduser(os.path.expandvars(models_dir))

    #enc = encoder.get_encoder(model_name, models_dir)
    container = []
    for i in range(180):
        intermediate = []
        for j in range(11):
            shape = enc.encode(' ' + top_ten.iloc[i, j])
            intermediate.append(shape)
        container.append(intermediate)
    return container


def Harry_sentences_no_capital(counter, Sent_num):

    harry_dir = str(os.path.dirname(os.path.abspath(__file__))
                    ) + '/look_ups_gpt-2/words_fmri.npy'
    namelist = ['Harry', 'Ron', 'Malfoy', 'Neville', 'Dumbledore',
                'Hermione', 'Potter', 'Weasley.', ' Potter',  'Potter,', 'broomstick', ' Potter,']
    harry = np.load(harry_dir)
    total = ''
    q = 0
    while q < Sent_num:
        booler = False
        first = True
        stringe = ''
        for i in range(harry.size):
            j = counter + i
            if not first:
                booler = any(letter.isupper() for letter in harry[j]) or booler
            else:
                booler = harry[j] in namelist or booler
            stringe = stringe + ' ' + harry[j]
            first = False
            if harry[j].__contains__('.'):
                counter = j+1
                break
        if not booler:
            q += 1
            total = total + ' ' + stringe
    return(total, counter)


def get_gpt_encodings(words, enc):
    #models_dir = str(os.path.dirname(
    #    os.path.abspath(__file__))) + '/../GPT2/gpt-2/models'

    #enc = encoder.get_encoder('774M', models_dir)
    gpt_embeddings = np.zeros(5)
    gpt_embeddings[0] = int(enc.encode(words[0])[0])#_single(words[0]))#"\u0120"+words[0]))
    gpt_embeddings[1] = int(enc.encode(words[1])[0])#_single(words[1]))#"\u0120"+words[1]))
    gpt_embeddings[2] = int(enc.encode(words[2])[0])#_single(words[2]))#"\u0120"+words[2]))
    gpt_embeddings[3] = int(enc.encode(words[3])[0])#_single(words[3]))#"\u0120"+words[3]))
    gpt_embeddings[4] = int(enc.encode(words[4])[0])#_single(words[4]))#"\u0120"+words[4]))

    return(gpt_embeddings)


def count_word_stem(word, sequence):
    #print ("Sequence", sequence)
    sequence = sequence.split()

    word_count = 0
    # sequence.count(word)
    #print("Word Stems")
    word_stem = porter.stem(word)
    #print(word_stem)
    for s_word in sequence:
        s_word_stem = porter.stem(s_word)
        #print(s_word_stem)
        if(s_word_stem == word_stem):
            word_count += 1

    #print("Word count")
    #print(word_count)
    return word_count

def lexical_diversity(text):
    return len(text) / len(set(text))

# A score function for how good our sentense is
def evaluate_quality(sequence, word, related_count, perplexity, guide):
    # we aim for one ocurance of the word,
    # Or two ocurances of related words
    # and low perplexity
    w_1 = 1
    w_2 = 2
    w_3 = 0.001   # Changed from 0.01!

    if(word == ""):
        quality_score = math.exp(-(w_1*(3) + w_3*perplexity))
        return quality_score

    quality_score = 0
    word_count = count_word_stem(word, sequence)
    #print("Quality score evaluation:")
    #print("Word count", word_count)

    if(word_count != 0) and guide:
        quality_score = math.exp(-(w_1*word_count + w_3*perplexity))
    #elif(related_count != 0):
    #    quality_score = math.exp(-(w_1 * (1) +
    #                               w_2 * abs(related_count - 2) +
    #                               w_3*perplexity))
    else:
        quality_score = math.exp(-(w_1*(2) + w_3*perplexity))

    #print("txt, quality_score, word_count, rel_count, ppl", sequence, quality_score, word_count, related_count, perplexity)

    #lex_div = lexical_diversity(sequence)

    return quality_score, word_count, #lex_div

