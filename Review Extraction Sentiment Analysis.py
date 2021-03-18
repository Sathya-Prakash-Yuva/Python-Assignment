# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 08:42:28 2021

@author: Lenovo
"""

import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
Headphones_reviews=[]

for i in range(1,10):
  ip=[]  
  url="https://www.amazon.in/255-Bluetooth-Wireless-Earphone-Immersive/dp/B07C2VJXP4/ref=sr_1_4?crid=T5TACWPGHEAM&dchild=1&keywords=bluetooth+headphones+wireless&qid=1609816147&sprefix=bluet%2Caps%2C374&sr=8-4="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
  Headphones_reviews=Headphones_reviews+ip
  
 with open("Headphones.txt","w",encoding='utf8') as output:
    output.write(str(Headphones_reviews)) 
    
ip_rev_string = " ".join(Headphones_reviews)  
ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string) 

ip_reviews_words = ip_rev_string.split(" ")
ip_reviews_words

stop_words = stopwords.words('english')

with open("E:\\stop.txt","r") as sw:
    stopwords = sw.read()
    
stopwords = stopwords.split("\n")    

temp = ["this","is","awsome","Data","Science"]
[i for i in temp if i not in "is"]

ip_reviews_words = [w for w in ip_reviews_words if not w in stopwords]
ip_reviews_words

ip_rev_string = " ".join(ip_reviews_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

with open("E:\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]  

with open("E:\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)

Headphone_unique_words = list(set(" ".join(Headphones_reviews).split(" ")))
Headphone_unique_words

