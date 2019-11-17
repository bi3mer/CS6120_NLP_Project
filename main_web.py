from flask import Flask, request
from flask import render_template
from webfunctions.TwitterDataHandler import TwitterDataHandler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/usertweets', methods=['POST'])
def usertweets():
    td = TwitterDataHandler()
    username = request.form.get("username")
    tweets_dict = td.scoreUser(username)
    return render_template("tweets.html",tweets=tweets_dict)


if __name__ == "__main__":
    app.run(debug=True)
