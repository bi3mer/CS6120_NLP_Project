from flask import Flask, request
from flask import render_template
from twitter import *

from Toxicity.Utility import log
from Toxicity import Model

import numpy as np
import os, ssl
import json

app = Flask(__name__)

def getTwitterKeys():
    with open(r"./twitter_api.template.json") as keyFile:
        keys = json.load(keyFile)

    consumer_key =keys['consumer_key']
    consumer_secret=keys['consumer_secret']
    token=keys['token']
    token_secret=keys['token_secret']

    return consumer_key,consumer_secret,token,token_secret


def getTweetsList(resp):
    tweet_text = dict()
    for i,t in enumerate(resp):
        tweet_text[i] = t.get('text','')
    return  tweet_text

def scoreTweets(tweets):
    model = Model()
    return np.mean([model.score(k) for k, v in tweets.items()])


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/usertweets', methods=['POST'])
def usertweets():
    username = request.form.get("username")
    print(username)
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context

    consumer_key,consumer_secret,token,token_secret = getTwitterKeys()
    t = Twitter(auth=OAuth(token, token_secret, consumer_key, consumer_secret))

    resp = t.statuses.user_timeline(screen_name=username)
    tweet_dict = getTweetsList(resp)
    score = scoreTweets(tweet_dict)
    tweet_dict['score']=(str(score))
    return render_template("tweets.html",tweets=tweet_dict)


if __name__ == "__main__":
    app.run(debug=True)
