from nltk import NaiveBayesClassifier as nbc
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
from flask import Flask,request, jsonify
import pandas as pd
import tweepy as tw
import json
import numpy as np

#############################################################################################
################################  TWEEPY SETUP  #############################################

consumer_key = 'KNY4Zvfpg3ZJw9HEXgYZgpEsV'
consumer_secret = 'omlOAjdQViBKm1IKVWQfa0xuPakw6qs8G3YgnVM796KuaEabVz'
access_token = '60773330-kwmBd0SRPws1b0xl9EqF6hqOGuzXp0gxU9dX1HKZi'
access_token_secret = 'Lxl9jZJIcS2QresKsaRaSEcxcCly5JVuA6gVBtrveY9Eh'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth)

#############################################################################################
################################  SENTIMENT SETUP  ##########################################

# pos.txt
with codecs.open('pos.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listpos=[e.strip() for e in lines]
del lines
f.close() 

# neg.txt
with codecs.open('neg.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listneg=[e.strip() for e in lines]
f.close() 

# neutral.txt
with codecs.open('neutral.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listneu=[e.strip() for e in lines]
f.close() 

pos1=['positive']*len(listpos)
neg1=['negative']*len(listneg)
neu1=['neutral']*len(listneu)

training_data = list(zip(listpos,pos1)) + list(zip(listneg,neg1)) + list(zip(listneg,neu1))
vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))
feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in training_data]
classifier = nbc.train(feature_set)

#############################################################################################

app = Flask(__name__)

def search_tweets():
    
    # Define the search term and the date_since date as variables
    search_words = request.args.get("keyword")

    new_search = search_words + "-filter:retweets" # Do not get retweet of tweets
    
    #collect tweets
    tweets = tw.Cursor(api.search_tweets, q = new_search,  lang = 'th' ).items(50)
    
    users_locs = [[ tweet.created_at, tweet.text, tweet.user.followers_count,tweet.retweet_count, tweet.favorite_count] for tweet in tweets]
    
    # To dateframe
    tweet_df = pd.DataFrame(data=users_locs, columns=['time_stamp', 'text', 'followers_count', 'retweet_count','favorite_count'])

    # To Json  
    result = tweet_df['text'].to_json(orient="records")
    parsed = json.loads(result)

    return parsed

#############################################################################################

def sentiment(text_tweet):
    test_sentence = text_tweet
    featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary}
    sentiment = classifier.classify(featurized_test_sentence)
    my_array = np.array([test_sentence, sentiment])

    return my_array
#############################################################################################

def sentiment_analyst():
    test = []
    for i in search_tweets():
        test.append(sentiment(i))
    df = pd.DataFrame(test, columns = ['text','sentiment'])
    result = df.to_json(orient="index")
    parsed = json.loads(result)
    return jsonify(parsed)

@app.route('/', methods=['GET'])
def home():
    return 'Hello World'

@app.route('/api/twitter-sentiments', methods=['GET'])
def get_api():
    return sentiment_analyst()

if __name__ == "__main__":
    app.run(debug=True)
