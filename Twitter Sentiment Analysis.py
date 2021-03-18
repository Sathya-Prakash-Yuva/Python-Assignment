# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:31:25 2021

@author: Lenovo
"""

import pandas as pd
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8

API_Key="mynXM0VAM6093khcmMhTGUopV"
API_secretKey="4dntFkhJpjd6TKLh0hrhjgCKU9153JuOhEK6d2HPFx7y08amHv"
Access_Token="1346409690826035203-qPBv4biwI8qZa6pMS3NG00jZf935m8"
Access_Tokensecret="XvphgbYZ0a4eZ3CttinNRGwYS8DeK3VQ0Sn4Z98UjgxKP"

auth = tweepy.OAuthHandler(API_Key,API_secretKey)
auth.set_access_token(Access_Token,Access_Tokensecret)
twetterApi = tweepy.API(auth, wait_on_rate_limit = True)

twitterAccount = "elonmusk"

tweets = tweepy.Cursor(twetterApi.user_timeline, 
                        screen_name=twitterAccount, 
                        count=None,
                        since_id=None,
                        max_id=None,
                        trim_user=True,
                        exclude_replies=True,
                        contributor_details=False,
                        include_entities=False
                        ).items(50);
                       
df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweet'])                       

df.head()
def cleanUpTweet(txt):
    # Remove mentions
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    # Remove hashtags
    txt = re.sub(r'#', '', txt)
    # Remove retweets:
    txt = re.sub(r'RT : ', '', txt)
    # Remove urls
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt)
    return txt

df['Tweet'] = df['Tweet'].apply(cleanUpTweet)

def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

df['Subjectivity'] = df['Tweet'].apply(getTextSubjectivity)
df['Polarity'] = df['Tweet'].apply(getTextPolarity)

df.head()

def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"
    
df['Score'] = df['Polarity'].apply(getTextAnalysis)    

df.head(50)

positive = df[df['Score'] == 'Positive']

print(str(positive.shape[0]/(df.shape[0])*100) + " % of positive tweets")

labels = df.groupby('Score').count().index.values

values = df.groupby('Score').size().values

plt.bar(labels, values)

for index, row in df.iterrows():
    if row['Score'] == 'Positive':
        plt.scatter(row['Polarity'], row['Subjectivity'], color="green")
    elif row['Score'] == 'Negative':
        plt.scatter(row['Polarity'], row['Subjectivity'], color="red")
    elif row['Score'] == 'Neutral':
        plt.scatter(row['Polarity'], row['Subjectivity'], color="blue")

plt.title('Twitter Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
# add legend
plt.show()

objective = df[df['Subjectivity'] == 0]

print(str(objective.shape[0]/(df.shape[0])*100) + " % of objective tweets")

words = ' '.join([tweet for tweet in df['Tweet']])
wordCloud = WordCloud(width=600, height=400).generate(words)

plt.imshow(wordCloud)
plt.show()

