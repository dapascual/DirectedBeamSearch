import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse
os.environ['TRANSFORMERS_CACHE']='./transformer_models'

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from utility_gpt import *
from perplexity import *
from nltk.stem import PorterStemmer


porter = PorterStemmer()


#import gensim.downloader as api
#word_vectors = api.load("glove-wiki-gigaword-300")





# Generate look_ups_gpt-2, containing
# words_fmri,
# Wordsets,
# Top Ten embedings,
# Converter table etc

print("Before")
if not files_exist():
    generate_look_ups(word_vectors)         ### WATCH OUT WITH CONVERT TABLE GLOVE
print("After")



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        #indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove )

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    return logits

def sample_sentence(text, this_sequence, tokenizer, model, guide_word_stem, glove_word, converter_table, weight, guide=False, prev_proba=1, top_k=0, top_p=0.9, temperature=1.):
    ## GPT2 - generate logits
    indexed_tokens = tokenizer.encode(text)
    indexed_this_seq = tokenizer.encode(this_sequence)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')
    
    # Predict all tokens
    #with torch.no_grad():
    outputs = model(tokens_tensor)
    logits = outputs.logits
    logits = logits[0, -1, :]/ temperature
    proba = F.softmax(logits, dim=-1)        
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    #print("Max logits after sampling: ", logits.shape, torch.max(logits))
    # Calculate cosine similarity
    if guide == True:
        sim = cosine_similarity(np.reshape(
                    glove_word, (1, -1)), converter_table)
                    
        sim = np.clip(np.squeeze(sim), a_min=0, a_max=None) #tf.square(sim)  
        sim = sim*sim
        #weight_ = 2*torch.max(logits[torch.isfinite(logits)]).item()        
        logits = logits + torch.tensor(sim*weight).cuda()

    logits = F.softmax(logits, dim=-1) 
    #logits = F.softmax(logits, dim=-1)
    predicted_index = torch.multinomial(logits, 1).item()
    
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
    pred_word = predicted_text.split()[-1]
    
    pred_word_stem = porter.stem(pred_word)
    
    if pred_word_stem == guide_word_stem:
        guide_next = False
    else:
        guide_next = guide
    
    return predicted_text, guide_next, prev_proba*proba[predicted_index], this_sequence

def sample_sentence_noguide(text, this_sequence, tokenizer, model, prev_proba=1, top_k=0, top_p=0.9, temperature=1.):
    ## GPT2 - generate logits
    indexed_tokens = tokenizer.encode(text)
    indexed_this_seq = tokenizer.encode(this_sequence)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')
    
    # Predict all tokens
    #with torch.no_grad():
    outputs = model(tokens_tensor)
    logits = outputs.logits
    logits = logits[0, -1, :]/ temperature
    proba = F.softmax(logits, dim=-1)        
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    #print("Max logits after sampling: ", logits.shape, torch.max(logits))
    # Calculate cosine similarity
    
    logits = F.softmax(logits, dim=-1) 
    #logits = F.softmax(logits, dim=-1)
    predicted_index = torch.multinomial(logits, 1).item()
    
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
    pred_word = predicted_text.split()[-1]  
    
    return predicted_text, prev_proba*proba[predicted_index], this_sequence

def conditional_language_generation(
    model_name='gpt2-large',
    word_set_index=0,
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.9,
    models_dir='models',
    constant=20,
    number_of_concurrent_sentences = 10,
    number_of_generated_sentences = 20,
    number_of_words_per_sentence = 5,
    number_of_beams = 3,
    number_keywords = 5,
    word_index=0,
    save_path='dummy.txt'    
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    :top_p=1 Top_p is the cummulative probability used for nucleus sampling. 1 means no nucleus sampling
    :constant: How much are anchors weighted
    :counter index of wordset which is currently evaluated
    :TODO ns.....
    """
    
    start_time = time.time()

    total_words = number_of_words_per_sentence*number_of_generated_sentences

    #GPT2  
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model.eval()
    
###################################
	## Load words

    length = number_of_words_per_sentence

    #word_set = np.load(str(os.path.dirname(os.path.abspath(
    #    __file__))) + '/look_ups_gpt-2/word_sets_num.npy')
    
    # line contains all sets of 5 worlds in it
    file1 = open(str(os.path.dirname(os.path.abspath(__file__))) +
                 "/look_ups_gpt-2/glove_words_param_eval/word_sets.txt", "r+")

    line = file1.readlines()
    keywords = list(line[word_index].strip().split(", "))
    print("Keywords: ", keywords)

    keywords_glove = np.load(str(os.path.dirname(
        os.path.abspath(__file__))) + '/look_ups_gpt-2/glove_words_param_eval/set_' + str(word_index) + '.npy') 

###################################
    from datetime import datetime
    now = datetime.now()
    #date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    text_file = open(save_path + '.txt', 'a+', encoding='utf8')
#str(os.path.dirname(os.path.abspath(__file__))) +"/results/result_subsequences_"+date_time+".txt", "a+", encoding='utf8')
    
    

    # prepare variables...
    np.random.seed(seed)

    weight = constant
    converter_table = np.load(str(os.path.dirname(
        os.path.abspath(__file__))) + '/look_ups_gpt-2/converter_table_glove.npy')  
    
    
    guidance_index = [0]*number_of_beams
    cum_quality_score = [0]*number_of_beams
    success_length =  [0]*number_of_beams
    online_probability = [1]*number_of_beams
    guide = [True]*number_of_beams
    full_text = ["It is"] * number_of_beams #"I think"
    
    for k in range(number_of_generated_sentences):      

        # Define guidance word and index for guidance in model and quality function     
        result_subsequences = []        
        for b in range(number_of_beams):

            beam_text = full_text[b]
            guidance_word = keywords[guidance_index[b]]
            print("Guidance: ", str(guidance_index[b]), guidance_word)
            guide_word_stem = porter.stem(guidance_word)
            
            perplexities = np.zeros((number_of_concurrent_sentences))
####################################### Generation loop ################################################
            for i in range(number_of_concurrent_sentences):
                
                context = beam_text
                proba = 1
                this_sequence = ""
                if guide[b]:
                    guide_next = True    
                    for j in range(number_of_words_per_sentence):
                        context, guide_next, proba, this_sequence = sample_sentence(context, this_sequence, tokenizer, model, 
                                guide_word_stem, keywords_glove[guidance_index[b]], converter_table, weight, guide_next, proba)
                else:   # Dont't guide                    
                    for j in range(number_of_words_per_sentence):
                        context, proba, this_sequence = sample_sentence_noguide(context, this_sequence, tokenizer, model, prev_proba=proba)
                        #context, _, proba, this_sequence = sample_sentence(context, this_sequence, tokenizer, model, 
                        #        "", None, converter_table, weight, False, proba)
                    
                proba = proba.item()
                perplexity = np.power(proba, (-1/length))
                
                counter_sim = 0
                quality_score, word_count = evaluate_quality(this_sequence, guidance_word, counter_sim, perplexity, guide[b])
                print("Beam, Guidance: ", b, guidance_word, str(guidance_index[b]), guide[b])
                print("txt, quality, wordC, ppl: ", this_sequence, cum_quality_score[b]+quality_score, word_count, perplexity)   

                result_subsequences.append(
                        [context, cum_quality_score[b]+quality_score, word_count, perplexity, proba, guidance_index[b], guide[b]])

                perplexities[i] = perplexity            

########################################################################################################
        
        result_subsequences = sorted(
            result_subsequences, key=lambda a_entry: a_entry[1], reverse=True)      

        for b in range(number_of_beams):
            full_text[b] = result_subsequences[b][0]
            cum_quality_score[b] = result_subsequences[b][1]
            guidance_index[b] = result_subsequences[b][5]
            guide[b] = result_subsequences[b][6]
            if result_subsequences[b][2] > 0: ## Word Count
                guidance_index[b] += 1
                if guidance_index[b] > number_keywords-1:
                    guide[b] = False
                    guidance_index[b] = 0
                    success_length[b] = k+1
                    
            n_words_counter = (k+1)*number_of_words_per_sentence
            online_probability[b] *= result_subsequences[b][4]
            online_perplexity = np.power(online_probability[b], (-1/n_words_counter))

            print(">>>>>>>>>>>>> BEAM: ", b)
            print("Guidance words: ", keywords)
            print("Current sentence: ", full_text[b])
            print("Guidance word, word count: ", keywords[guidance_index[b]], result_subsequences[b][2])
            print("Current perplexity, cumulative quality: ", online_perplexity, cum_quality_score[b])        

                  
        '''
        text_file.write("\nBest 10 next subsequences: \n")
        for result_subsequence in result_subsequences:
            text_file.write(result_subsequence[0] + "\n Perplexity:" +
                            str(result_subsequence[2]) + "\n Quality Score: " +
                            str(result_subsequence[1]) + "\n\n")

        text_file.write("\n\n\n\n")
        '''
    #######################################
    # final evaluation
    #######################################
    '''
    if guidance_index[b] > 4:
        for j in range(total_words - n_words_counter):
            full_text, _, _, _ = sample_sentence(full_text, "", tokenizer, model, 
                        "", None, converter_table, None, False, 1)
    '''
    for b in range(number_of_beams):    
        if guide[b]:
            success_length[b] = 0

    # Success rate
    target_words = number_keywords
    target_count = 0
    for i in range(number_keywords):
        #if keywords[i] in plugAndPlayText:
        if count_word_stem(keywords[i], full_text[0]) > 0:
            target_count += 1
            
    success_rate = target_count/target_words

    # GPT (1) perplexity
    gpt_1_perplexity = gpt_1_perplexity_score(full_text[0])
    print("------------------------------------------------------------------------------")
    print("FINAL TEXT: ")
    print(full_text[0])
    print("Success rate, success length, perplexity: ", success_rate, success_length[0], gpt_1_perplexity)
    # Time measurement
    end_time = time.time()
    time_needed = end_time - start_time

    # Save evaluations

    # declare evaluations
    evaluation = {
        "final_sequence: ": full_text[0],
        "keywords": keywords,
        #"online_perplexity": online_perplexity[0],
        "gpt_1_perplexity": gpt_1_perplexity,
        "success_rate": success_rate,
        "number_of_concurent_sentences": number_of_concurrent_sentences,
        "number_of_generated_sentences": number_of_generated_sentences,
        "number_of_words_per_sentence": number_of_words_per_sentence,
        "total_words": total_words,
        "top_k": top_k,
        "top_p": top_p,
        "model_name": model_name,
        "constant": constant,
        "time_needed": time_needed,
        "success_length": success_length[0]
    }

    # To text file
    text_file.write("Keywords: \n")
    for word in keywords:
        text_file.write(word + " ")

    text_file.write("\n\n")

    text_file.write("Final sequence: \n\n")
    text_file.write(full_text[0])
    #text_file.write("\n\nperplexity: ")
    #text_file.write(str(online_perplexity[0]))
    text_file.write("\nSuccess_rate: " + str(success_rate))
    text_file.write("\nPerplexity: " + str(gpt_1_perplexity))
    text_file.write("\nTime_needed: " + str(time_needed))
    text_file.write("\nSuccess_length: " + str(success_length[0]))
    text_file.write("\n\n")

    text_file.close()

    print("END: ", keywords)

    return evaluation


if __name__ == '__main__':

    # Get constant defined in run_gpt2.sh
    parser = argparse.ArgumentParser()
    parser.add_argument('-weight', type=float, default=20)
    parser.add_argument('-n_concurrent_sentences', type=int, default=4)
    parser.add_argument('-n_generated_sentences', type=int, default=20)
    parser.add_argument('-n_words_per_sentence', type=int, default=5)
    parser.add_argument('-n_beams', type=int, default=5)
    parser.add_argument('-n_word_files', type=int, default=50)
    args = parser.parse_args()

    word_set_index = 0
    weight = args.weight
    number_of_concurrent_sentences = args.n_concurrent_sentences
    number_of_generated_sentences = args.n_generated_sentences
    number_of_words_per_sentence = args.n_words_per_sentence
    number_of_beams = args.n_beams
    n_word_files = args.n_word_files

    save_file = 'result_w_'+str(weight)+'_nBeams_'+str(number_of_beams)+'_nConcSent_'+str(number_of_concurrent_sentences)+'_nGenSent_'+str(number_of_generated_sentences)+'_nWordsPerSent_'+str(number_of_words_per_sentence)
    sub_folder = 'param_evaluation/'
    save_path = 'results/' + sub_folder + save_file

    all_results = np.zeros([n_word_files, 10, 4])
    # For every concept word set
    for j in range(n_word_files):
        for i in range(10):
            # Interact model => generate sentenses and evaluate them
            results = conditional_language_generation(constant=weight,
                                                      number_of_concurrent_sentences=number_of_concurrent_sentences,
                                                      number_of_generated_sentences=number_of_generated_sentences,
                                                      number_of_words_per_sentence=number_of_words_per_sentence,
                                                      number_of_beams = number_of_beams,
                                                      word_index=j,  
                                                      save_path=save_path           
                                                      )
            all_results[j][i][0] = results["gpt_1_perplexity"]
            all_results[j][i][1] = results["time_needed"]
            all_results[j][i][2] = results["success_rate"]
            all_results[j][i][3] = results["success_length"]        
    
    np.save(save_path, all_results)

    
    
