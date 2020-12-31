import math
import torch
import os
os.environ['TRANSFORMERS_CACHE']='.'

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')


def gpt_1_perplexity_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor(
        [tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss, logits = model(tensor_input, labels=tensor_input)[:2]

    return math.exp(loss)


#print(gpt_1_perplexity_score("I think investors will look at our results and the direction that we are taking and they will feel better about the direction that we are taking. And so, I don't see a need for them to do any further adjustments in their expectations. We believe this business continues to have significant upside potential over the medium term. So, I think investors should look forward to a positive earnings announcement today. And they may also consider selling shares of Exxon Mobil Corp., Inc., (NYSE:XOM) on short sale basis during open-outcry period if they wish as we expect higher earnings guidance for fiscal year 2014 and full year 2014 than last quarter, plus lower commodity prices could impact operating results materially through Q4'14 or Q1'15 when most of the refining sector would be operating in normal range versus 2013 volumes. If you would like additional information on Exxon Mobil Corp., Incorporated visit: www.exxonmobilcorpinc.com 1Q14 Guidance"))
