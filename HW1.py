import pandas as pd
import numpy as np
from scipy import spatial
import re

sentences = map(lambda x: x.strip().lower(), sents)
num_sentences = len(sentences)
sentence_tokens = map(lambda s: filter(lambda x: x!= '', re.split('[^a-z]', s)), sentences)
words = {}
curr_index = 0
for sentence_token in sentence_tokens:
    for token in sentence_token:
        if token not in words:
            words[token] = curr_index
            curr_index += 1
num_words = len(words)
matrix = np.zeros((num_sentences, num_words))
for i in range(num_sentences):
    tokens = sentence_tokens[i]
    for token in tokens:
        matrix[i][words[token]] += 1
distances = {}

first_sentence_metric = matrix[0, :]
for i in range(num_sentences):
    cmp_sentence_metric = matrix[i, :]
    
    distances[i] = spatial.distance.cosine(first_sentence_metric, cmp_sentence_metric)
    distances_df = pd.DataFrame.from_dict(distances, orient = 'index')
distances_df.columns = ['distance']
distances_df['sentence'] = map(lambda x: sentences[x], distances_df.index.values)

distances_df.sort('distance')[:4]
