import pandas as pd

df = pd.read_csv('tweets.csv')

def tod_converter(timestamp):
    converter = {'morning': (5, 13),
                 'afternoon': (13, 16),
                 'evening': (16, 23)}
    
    hour = timestamp.hour
    
    for tod in converter:
        if hour in range(*converter[tod]):
            return tod
    return 'night'

df['created_at'] = pd.to_datetime(df['created_at'])
df['time_of_day'] = df['created_at'].apply(tod_converter)

#TASK 1

def parse_handle(tweet):
    handle = None
    handle = tweet[tweet.find('@'):tweet.find(':')]
    return handle
df['handle'] = df['tweet'].apply(parse_handle)

import re
def delete_handle(tweet):
    no_handle = re.sub("^@.*?:", '', tweet)
    return no_handle
df['tweet'] = df['tweet'].apply(delete_handle)

#TASK 2
def count_mentions(tweet):      
    m = len(re.findall(r" @", tweet))
    return m

df['num_mentions'] = df['tweet'].apply(count_mentions)

#TASK 3
def months(timestamp):

    month = timestamp.month
    return month

df['created_at'] = pd.to_datetime(df['created_at'])
df['month'] = df['created_at'].apply(months)

df2 = df.sort_values(by='month')
tweets_per_month = df2['month'].value_counts(sort=False).rename_axis('month').to_frame('tweets_per_month')
print(tweets_per_month)

tweets_per_month.plot.bar()

df['rt_ratio'] = df['retweets'] / df['likes']
df.sort_values(by='likes')
df['rt_ratio'][20602]

#TASK 3
import numpy as np
from numpy import inf
df['rt_ratio'].replace(np.inf, 0, inplace=True)
print(df)

df['rt_ratio'][20602]

#TASK 4
num_tw = df.groupby('handle').agg({'likes': 'mean','tweet': 'size'})

new_df = num_tw[num_tw.tweet > 500]
print(new_df)

new_df = new_df.sort_values(by='likes', ascending=False)
top5 = new_df[:5]
print(top5)

tops = top5.index.tolist()
print(tops)

top5_df = df[df.handle.isin(t)]
top5_df = top5_df.groupby(['handle', 'tweet']).first()
print(top5_df)

new_df_more30 = top5_df[top5_df.likes > 30]
print(new_df_more30)

#TASK 5
time_of_daydf = new_df_l.groupby(['handle', 'time_of_day']).agg({'rt_ratio':'mean'})
print(time_of_daydf)

time_of_daydf.unstack().plot.barh()

#TASK 6
import nltk
from nltk import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
def count_sentiment_score(tweet):
    def zer_dev(a,b):
        try:
            return a/b
        except ZeroDivisionError:
            return 0
    positive_words = open('positive-words.txt').readlines()
    negative_words = open('negative-words.txt').readlines()
    positive_w = []
    negative_w = []
    for i in positive_words:
        i = i.strip('\n')
        positive_w.append(i)
    for i in negative_words:
        i = i.strip('\n')
        negative_w.append(i)
    tweet = str(tweet).lower()
    tweet = re.sub('[^a-zA-Z]+',' ', tweet).strip()
    score = 0
    pos = []
    neg = []    
    tok_sent = nltk.word_tokenize(tweet)
    for word in tok_sent:
        word = wnl.lemmatize(word)
        if word in positive_w:
            pos.append(word)
        elif word in negative_w:
            neg.append(word)

    pos_neg_dif = (len(pos)-len(neg))
    score = zer_dev(pos_neg_dif, len(tok_sent))
    return score
df['sentiment']=df['tweet'].apply(count_sentiment_score)
print(df)

a = df.sentiment.quantile(0.1)#нег
b = df.sentiment.quantile(0.9)#поз
a1 = df.loc[df['sentiment'] <= a, 'tweet']
b1 = df.loc[df['sentiment'] >= b, 'tweet']
a2 = a1.tolist() 
b2 = b1.tolist()

for tweet in b2:
    print('TWEET: \n')
    print(tweet)
    print()
