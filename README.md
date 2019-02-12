# Latent Dirichlet allocation and timeseries analysis for summarization of live TV shows via Twitter
---
#### Watching TV is usually accompanied with comments about the content. We tend to address these comments to nearby people or online friends. In this empirical study we retrieved 30k Twitter status updates during popular TV talk shows. Topic modeling analysis allows us to separate themes and, eventually, summarize long TV broadcasts automatically. 

| Tweets volume change during talkshow *Enikos* (18/4/2016)  | Top LDA topics during talkshow *Anatropi* (12/4/2016) |
| ------------- | ------------- |
| ![PICTURE](https://github.com/sdimi/tv-show-summarization-twitter/blob/master/figures/timeseries.PNG)  | ![PICTURE](https://github.com/sdimi/tv-show-summarization-twitter/blob/master/figures/topics.PNG)  |

We provide the [source code](https://github.com/sdimi/tv-show-summarization-twitter/blob/master/tv%20summarization.py) for our analysis, the [pdf report](https://github.com/sdimi/tv-show-summarization-twitter/blob/master/report.pdf) and a [Jupyter Notebook](https://github.com/sdimi/tv-show-summarization-twitter/blob/master/jupyter%20notebook.ipynb) with both input and output. You can also watch the presentation [here](https://speakerdeck.com/sdimi/topic-modeling-and-summarization-of-live-tv-shows-via-twitter).

---

#### Algorithm walkthrough
1. Read tweets from [tweepy](https://github.com/tweepy/tweepy) retrieved hashtag json/txt.
2. Load to pandas DataFrame and keep only relevant columns (tweets and timestamp in our case).
3. (Optional) Transform timestamp to local timezone (Athens time in our case, advise [pytz](http://pytz.sourceforge.net/) to adjust).
4. Count tweet volume per minute.
5. Plot time series.
6. (Optional) Remove stopwords, find most frequent tokens (Greek stopwords in our case but every NLTK supported language works).
7. LDA Preprocessing, words occurring in only one document or in at least 95% of the documents are removed.
8. Document Term Matrix structure transform.
9. Obtain the words with high probabilities.
10. Obtain the feature names.
11. Print LDA topics, assign topics to each tweet.

---
#### Dependencies 
* Python 2.7+
* Scikit-learn
* Pandas
* Numpy
* Vincent
* NLTK
* LDA
