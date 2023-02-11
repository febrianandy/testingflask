from flask import Flask, request, render_template, json, jsonify
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn
from textstat.textstat import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# %matplotlib inline

# Label "0" = Non Hate Speech
# Label "1" = Hate Speech

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/gaussion", methods=['POST', 'GET'])
def gaussion():
    dataset = pd.read_csv('filekuu.csv')
    dataset.head()
    dataset['text length'] = dataset['Tweet'].apply(len)
    ## 1. Removal of punctuation and capitlization
    ## 2. Tokenizing
    ## 3. Removal of stopwords
    ## 4. Stemming

    stopwords = nltk.corpus.stopwords.words("english")

    #extending the stopwords to include other words used in twitter such as retweet(rt) etc.
    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)
    stemmer = PorterStemmer()
    # collecting only the tweets from the csv file into a variable name tweet
    tweet=dataset.Tweet
    def preprocess(tweet):  
        
        # removal of extra spaces
        regex_pat = re.compile(r'\s+')
        tweet_space = tweet.str.replace(regex_pat, ' ')

        # removal of @name[mention]
        regex_pat = re.compile(r'@[\w\-]+')
        tweet_name = tweet_space.str.replace(regex_pat, '')
        
        # removal of RT
        regex_pat = re.compile(r'RT')
        tweet_rt = tweet_name.str.replace(regex_pat,'')

        # removal of links[https://abc.com]
        giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        tweets = tweet_rt.str.replace(giant_url_regex, '')
        
        # removal of punctuations and numbers
        punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
        
        # remove whitespace with a single space
        newtweet=punc_remove.str.replace(r'\s+', ' ')
        
        # remove leading and trailing whitespace
        newtweet=newtweet.str.replace(r'^\s+|\s+?$','')
        
        # replace normal numbers with numbr
        newtweet=newtweet.str.replace(r'\d+(\.\d+)?','numbr')
        
        # removal of capitalization
        tweet_lower = newtweet.str.lower()
        
        # tokenizing
        tokenized_tweet = tweet_lower.apply(lambda x: x.split())
        
        # removal of stopwords
        tokenized_tweet=  tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])
        
        # stemming of the tweets
        tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
        
        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
            tweets_p= tokenized_tweet
        
        return tweets_p

    processed_tweets = preprocess(tweet)   
    dataset['Processing']=preprocess(dataset['Tweet'])
    dataset.head()

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)

    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(dataset['Processing'] )
    tfidf
        # If you don't specify the random_state in the code, 
    # then every time you run(execute) your code a new random value is generated 
    # and the train and test datasets would have different values each time.
    X = tfidf
    y = dataset['Label'].astype(int)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X.toarray(), y, random_state=42, test_size=0.2)
    nb=GaussianNB()
    nb.fit(X_train_tfidf,y_train)
    y_preds = nb.predict(X_test_tfidf)
    
    inp = request.form['kata']
    hasil = tfidf_vectorizer.transform([inp]).toarray()
    pred = nb.predict(hasil)
    accuracy = nb.score(X_test_tfidf, y_test)
    if pred == 1:
        return jsonify({'hate_speech': 'Hate Speech', 'accuracy': accuracy})
    else:
        return jsonify({'hate_speech': 'Not Hate Speech', 'accuracy': accuracy})

 


@app.route("/gausion_bernolui", methods=['POST', 'GET'])
def gausion_bernolui():
    dataset = pd.read_csv('filekuu.csv')
    dataset.head()
    dataset['text length'] = dataset['Tweet'].apply(len)
    ## 1. Removal of punctuation and capitlization
    ## 2. Tokenizing
    ## 3. Removal of stopwords
    ## 4. Stemming

    stopwords = nltk.corpus.stopwords.words("english")

    #extending the stopwords to include other words used in twitter such as retweet(rt) etc.
    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)
    stemmer = PorterStemmer()
    # collecting only the tweets from the csv file into a variable name tweet
    tweet=dataset.Tweet
    def preprocess(tweet):  
        
        # removal of extra spaces
        regex_pat = re.compile(r'\s+')
        tweet_space = tweet.str.replace(regex_pat, ' ')

        # removal of @name[mention]
        regex_pat = re.compile(r'@[\w\-]+')
        tweet_name = tweet_space.str.replace(regex_pat, '')
        
        # removal of RT
        regex_pat = re.compile(r'RT')
        tweet_rt = tweet_name.str.replace(regex_pat,'')

        # removal of links[https://abc.com]
        giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        tweets = tweet_rt.str.replace(giant_url_regex, '')
        
        # removal of punctuations and numbers
        punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
        
        # remove whitespace with a single space
        newtweet=punc_remove.str.replace(r'\s+', ' ')
        
        # remove leading and trailing whitespace
        newtweet=newtweet.str.replace(r'^\s+|\s+?$','')
        
        # replace normal numbers with numbr
        newtweet=newtweet.str.replace(r'\d+(\.\d+)?','numbr')
        
        # removal of capitalization
        tweet_lower = newtweet.str.lower()
        
        # tokenizing
        tokenized_tweet = tweet_lower.apply(lambda x: x.split())
        
        # removal of stopwords
        tokenized_tweet=  tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])
        
        # stemming of the tweets
        tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
        
        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
            tweets_p= tokenized_tweet
        
        return tweets_p

    processed_tweets = preprocess(tweet)   
    dataset['Processing']=preprocess(dataset['Tweet'])
    dataset.head()

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)

    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(dataset['Tweet'] )
    tfidf
     # If you don't specify the random_state in the code, 
    # then every time you run(execute) your code a new random value is generated 
    # and the train and test datasets would have different values each time.
    X = tfidf
    y = dataset['Label'].astype(int)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X.toarray(), y, random_state=42, test_size=0.2)
    nb=BernoulliNB()
    nb.fit(X_train_tfidf,y_train)
    y_preds = nb.predict(X_test_tfidf)
   
    
    inp = request.form['kata']
    hasil = tfidf_vectorizer.transform([inp]).toarray()
    pred = nb.predict(hasil)
    accuracy = nb.score(X_test_tfidf, y_test)
    if pred == 1:
        return jsonify({'hate_speech': 'Hate Speech', 'accuracy': accuracy})
    else:
        return jsonify({'hate_speech': 'Not Hate Speech', 'accuracy': accuracy})

@app.route("/gaussian_multinominal", methods=['POST', 'GET'])
def gaussian_multinominal():
    dataset = pd.read_csv('filekuu.csv')
    dataset.head()
    dataset['text length'] = dataset['Tweet'].apply(len)
    ## 1. Removal of punctuation and capitlization
    ## 2. Tokenizing
    ## 3. Removal of stopwords
    ## 4. Stemming

    stopwords = nltk.corpus.stopwords.words("english")

    #extending the stopwords to include other words used in twitter such as retweet(rt) etc.
    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)
    stemmer = PorterStemmer()
    # collecting only the tweets from the csv file into a variable name tweet
    tweet=dataset.Tweet
    def preprocess(tweet):  
        
        # removal of extra spaces
        regex_pat = re.compile(r'\s+')
        tweet_space = tweet.str.replace(regex_pat, ' ')

        # removal of @name[mention]
        regex_pat = re.compile(r'@[\w\-]+')
        tweet_name = tweet_space.str.replace(regex_pat, '')
        
        # removal of RT
        regex_pat = re.compile(r'RT')
        tweet_rt = tweet_name.str.replace(regex_pat,'')

        # removal of links[https://abc.com]
        giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        tweets = tweet_rt.str.replace(giant_url_regex, '')
        
        # removal of punctuations and numbers
        punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
        
        # remove whitespace with a single space
        newtweet=punc_remove.str.replace(r'\s+', ' ')
        
        # remove leading and trailing whitespace
        newtweet=newtweet.str.replace(r'^\s+|\s+?$','')
        
        # replace normal numbers with numbr
        newtweet=newtweet.str.replace(r'\d+(\.\d+)?','numbr')
        
        # removal of capitalization
        tweet_lower = newtweet.str.lower()
        
        # tokenizing
        tokenized_tweet = tweet_lower.apply(lambda x: x.split())
        
        # removal of stopwords
        tokenized_tweet=  tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])
        
        # stemming of the tweets
        tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
        
        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
            tweets_p= tokenized_tweet
        
        return tweets_p

    processed_tweets = preprocess(tweet)   
    dataset['Processing']=preprocess(dataset['Tweet'])
    dataset.head()

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)

    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(dataset['Tweet'] )
    tfidf
        # If you don't specify the random_state in the code, 
    # then every time you run(execute) your code a new random value is generated 
    # and the train and test datasets would have different values each time.
    X = tfidf
    y = dataset['Label'].astype(int)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X.toarray(), y, random_state=42, test_size=0.2)
    nb=MultinomialNB()
    nb.fit(X_train_tfidf,y_train)
    y_preds = nb.predict(X_test_tfidf)
    
    
    inp = request.form['kata']
    hasil = tfidf_vectorizer.transform([inp]).toarray()
    pred = nb.predict(hasil)
    accuracy = nb.score(X_test_tfidf, y_test)
    if pred == 1:
        return jsonify({'hate_speech': 'Hate Speech', 'accuracy': accuracy})
    else:
        return jsonify({'hate_speech': 'Not Hate Speech', 'accuracy': accuracy})


if __name__ == "__main__":
    app.run(debug=True)
