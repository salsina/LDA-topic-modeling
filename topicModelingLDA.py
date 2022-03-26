import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("sample50000.csv")
data = data[data['language'] != 'language']

import re
from hazm import *
stopword_dict = {sw:0 for sw in stopwords_list()}
stemmer = Stemmer()
lemmatizer = Lemmatizer()
tweets = data['tweet']
clean_tweets = []

# tokenizing and cleaning the tweets
for tweet in tweets:
    tweet = re.sub(r'[@|#][A-Za-z-0-9_]*|(https:[A-Za-z-0-9_:/.]*)', '', tweet)
    temp_tweet = tweet.strip().split()
    final_tweet = []
    
    for word in temp_tweet:
        
        word = re.sub(r'[؟|!|،|.|؛|:]', '', word)
        if word in  ['؟',',','«','»','(',')','[',']','{','}','-','','؛','\r','\n']:
                continue
        if word in stopword_dict:
            continue
        word = stemmer.stem(word)
        word = lemmatizer.lemmatize(word)
        if word != '':
            final_tweet.append(word) 
    
    clean_tweets.append(final_tweet)
    

# finding the most frequent words in tweets
corpus = tweets
vec = CountVectorizer().fit(corpus)
bag_of_words = vec.transform(corpus)
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
most_frq_words = words_freq[:300]

for i in range(len(most_frq_words)):
    most_frq_words[i] = most_frq_words[i][0]

# removing frequent words in tweets
temp_clean_tweets = []
for i in range(len(clean_tweets)):
    temp = []
    for j in range(len(clean_tweets[i])):
        if clean_tweets[i][j] not in most_frq_words:
            temp.append(clean_tweets[i][j])
    temp_clean_tweets.append(temp)
clean_tweets = temp_clean_tweets

# building the LDA model
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

id2word = corpora.Dictionary(clean_tweets)
texts = clean_tweets

corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# writing the seperated words in each topic in a file
with open('fname', "a", encoding="utf-8") as f:
    f.write(str(lda_model.print_topics()[0]))
    f.write("\n")
    f.write(str(lda_model.print_topics()[1]))
    f.write("\n")
    f.write(str(lda_model.print_topics()[2]))
    f.write("\n")
    f.write(str(lda_model.print_topics()[3]))
    f.write("\n")
    f.write(str(lda_model.print_topics()[4]))
    f.write("\n")
    


doc_lda = lda_model[corpus]
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=clean_tweets, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)