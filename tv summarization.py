# Author: Dimitris Spathis <sdimitris@csd.auth.gr>
# Social Media, 2nd Semester, csd auth master program

#Analysis of Twitter reaction on TV talkshows

import json
import pandas as pd
from pandas.tseries.resample import TimeGrouper
from pandas.tseries.offsets import DateOffset

#load tweets txt, read every line
#data retrieved by tweepy, json saved as txt
tweets_data_path = 'data.txt' 
 
tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

#debug tweets number
print (len(tweets_data))        

#insert tweets to a Pandas dataframe
tweets = pd.DataFrame()

#get ['columns']

tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))  
#list (map..) in order to parse inside the array
tweets['created_at'] = list(map(lambda tweet: tweet['created_at'], tweets_data))

#debug show first tweets
tweets.head()

#save dataframe to csv
tweets.to_csv('teliko.csv')

#transform GMT timezone to Europe Athens
#creates new timestamp on the left of the dataframe
tweets['created_at'] = pd.to_datetime(pd.Series(tweets['created_at']))
tweets.set_index('created_at', drop=False, inplace=True)
tweets.index = tweets.index.tz_localize('GMT').tz_convert('Europe/Athens')
tweets.index = tweets.index - DateOffset(hours = 24) #isws auto prepei na ginei 24 hours
tweets.index


#created_at timeseries in a per minute minute format
tweets1m = tweets['created_at'].resample('1t').count()

#average tweets per minute
avg = tweets1m.mean()

#plot tweets timeseries
import vincent
vincent.core.initialize_notebook()
area = vincent.Area(tweets1m)
area.colors(brew='Spectral')
area.axis_titles(x='time', y='tweets')
area.display()

#find most frequent tokens with NLTK
#remove english stopwords
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist

#added "greek" file at /nltk_data/corpora/stopwords

#Combined the following files:
#file source #1 https://code.grnet.gr/projects/gss/repository/revisions/d59fbcd2f0cd/entry/solr/conf/stopwords.txt
#file source #2 http://grepcode.com/file/repo1.maven.org/maven2/gr.skroutz/elasticsearch-skroutz-greekstemmer/0.0.4/org/elasticsearch/index/analysis/stopwords.txt

#stop = stopwords.words('greek')
stop = stopwords.words('english')
text = tweets['text']

#tokenise
tokens = []
for txt in text.values:
    tokens.extend([t.lower().strip(":,.") for t in txt.split()])

filteredtokens = [w for w in tokens if not w in stop]

#compute frequency distribution
freqdist = nltk.FreqDist(filteredtokens)
#find 100 most frequent words
freqdist = freqdist.most_common(100)
freqdist


#LDA topic modelling 
# http://stackoverflow.com/questions/32055071/lda-topic-modeling-input-data
# https://ariddell.org/lda.html

import numpy as np
import lda
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tokens = {}

for i in range(len(filteredtokens)):
    tokens[i] = filteredtokens[i]

len(tokens)

#We use a few heuristics to strip accents (ά->α).
#Words occurring in only one document or in at least 95% of the documents are removed.

# More options:
# stop_words={'enikos','rt','dora_bakoyannis','https'})
# stop_words= stopwords.words('greek')

tf = CountVectorizer(strip_accents='unicode', max_df=0.95, min_df=2,stop_words=stopwords.words('greek'))
tfs1 = tf.fit_transform(tokens.values())
num = 8
model = lda.LDA(n_topics=num, n_iter=500, random_state=1)

#Document Term Matrix structure
model.fit_transform(tfs1)

#Obtain the words with high probabilities
topic_word = model.topic_word_

#Obtain the feature names
vocab = tf.get_feature_names()

#choose how many words per topic
n_top_words = 8
for i, tokens in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(tokens)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


#get the topic of each tweet (text[i])
doc_topic = model.doc_topic_    

for i in range(10):
     print("{} (top topic: {})".format(text[i], doc_topic[i].argmax()))







