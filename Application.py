import json
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import nltk
import requests

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask_cors import CORS
nltk.data.path.append('./nltk_data/')
app = Flask(__name__)
api = Api(app)
CORS(app)

sid = SentimentIntensityAnalyzer()

@app.route('/')
def index():
	return 'Yo, it\'s working!'


class RandmMeta(Resource):
    def get(self):
        d = {}
        with open('output.csv', 'r', encoding='utf8', errors='ignore') as f:
            reader = csv.reader(f, dialect='excel')
            # reader  = f.read();
            d = {'data': [{'score': i[0], 'sentence': i[1]} for i in reader]}
        return random.choice(d['data'])


class NLTKMeta(Resource):
    def post(self):
        start = time.time()
        json_data = request.get_json(force=True)
        sentence = json_data["sentence"]
        polarity = sid.polarity_scores(sentence)
        end = time.time()
        response_time = end - start
        return {'data': polarity,'responseTime':response_time}


class MicrosoftMeta(Resource):
    def post(self):
        start = time.time()
        microsoft = Microsoft()
        json_data = request.get_json(force=True)
        sentence = json_data["sentence"]
        result = microsoft.getScore(sentence)
        end = time.time()
        response_time = end - start
        return {'data': result, 'responseTime':response_time}


class WatsonMeta(Resource):
    def post(self):
        start = time.time()
        watson = Watson()
        json_data = request.get_json(force=True)
        sentence = json_data["sentence"]
        result = watson.getScore(sentence)
        end = time.time()
        response_time = end -start
        return {'data': result, 'responseTime':response_time}


class GoogleMeta(Resource):
    def post(self):
        start = time.time()
        google = Google()
        json_data = request.get_json(force=True)
        sentence = json_data["sentence"]
        result = google.getScore(sentence)
        end = time.time()
        response_time = end -start
        return {'data': result, 'responseTime':response_time}


api.add_resource(NLTKMeta, '/api/opensource/score/')
api.add_resource(MicrosoftMeta, '/api/azure/score/')
api.add_resource(WatsonMeta, '/api/watson/score/')
api.add_resource(GoogleMeta, '/api/google/score/')
api.add_resource(RandmMeta, '/api/random/sentence/')


class Google(object):
    url = "https://language.googleapis.com/v1/documents:analyzeEntitySentiment?key=AIzaSyBwpMG6Z0BKMnxKhSjFmPcVBz1ga_HIl6w"
    headers = {"Content-Type": "application/json"}
    data = {"document": {"type": "PLAIN_TEXT", "content": ""}, "encodingType": "UTF8"}

    def getScore(self, sentence):
        self.data["document"]["content"] = sentence
        r = requests.post(self.url, headers=self.headers, data=json.dumps(self.data))
        return r.json()


# vendors
class Microsoft(object):
    url = "https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment"
    headers = {'Ocp-Apim-Subscription-Key': '4f3d38b315c74e578e5a70777c0c92ad',
               'Content-Type': 'application/json'}
    data = {"documents": [{"id": "1", "text": ""}]}

    def getScore(self, sentence):
        self.data['documents'][0]["text"] = sentence
        r = requests.post(self.url, headers=self.headers, data=json.dumps(self.data))
        return r.json()


# IBM BLuemix
class Watson(object):
    url = "https://gateway.watsonplatform.net/natural-language-understanding/api/v1/analyze?version=2017-02-27&text=I%20still%20have%20a%20dream%2C%20a%20dream%20deeply%20rooted%20in%20the%20American%20dream%20%E2%80%93%20one%20day%20this%20nation%20will%20rise%20up%20and%20live%20up%20to%20its%20creed%2C%20%22We%20hold%20these%20truths%20to%20be%20self%20evident%3A%20that%20all%20men%20are%20created%20equal.&features=sentiment"
    headers = {'Authorization': "Basic MjJmMzQ2M2YtYThhZC00MGE1LTk0NzAtMmFlZDc3ZTcxNjdiOkFHS244S3I1Rmhmdw==",
               'Content-type': 'application/json'}
    data = {"text": ""}

    def getScore(self, sentence):
        self.data["text"] = sentence
        r = requests.post(self.url, headers=self.headers, data=json.dumps(self.data))
        return r.json()


if __name__ == '__main__':
    app.run()
