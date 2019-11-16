from flask import Flask, request
from flask import render_template
from twitter import *

from Toxicity.Utility import log
from Toxicity import Model

import numpy as np

app = Flask(__name__)


def getTweetsList(resp):
    tweet_text = dict()
    for i,t in enumerate(resp):
        tweet_text[i] = t.get('text','')
    return  tweet_text

def scoreTweets(tweets):
    scores = []
    model = Model()
    for k,text in tweets.items():
        scores.append(model.score(k))

    return np.mean(scores)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/usertweets', methods=['POST'])
def usertweets():
    username = request.form.get("username")
    b = {0: 0, 1: 1, 2: 2, 3: 3}
    print(username)
    import os, ssl
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context


    consumer_key = '0cZwgY2KG9kCJbN7vwlA1jNSF'
    consumer_secret = 'aP8SVyw3kfiYZPGbTwOl6t5n8gjIIvFVYOzQ7IwSWU8GtPPn6T'
    token = "3300316594-XzDEJ4Wtlv8eDgWQwDzqf9A6AaYXbfzq6m3sw43"
    token_secret = "PVMYQmpC44NVRir8db9yHBjHxk5YkMj9YWjIizqh5GXDh"
    t = Twitter(auth=OAuth(token, token_secret, consumer_key, consumer_secret))
    resp = t.statuses.user_timeline(screen_name=username)
    print(resp)
    tweet_dict = getTweetsList(resp)
    score = scoreTweets(tweet_dict)
    tweet_dict['score']=(str(score))
    return render_template("tweets.html",tweets=tweet_dict)


if __name__ == "__main__":
    app.run(debug=True)
