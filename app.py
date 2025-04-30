from flask import Flask, render_template, request, jsonify
from model import loadModel, nextWordPrediction
import os

app = Flask(__name__)

modelVocab, model = loadModel(os.path.join(app.root_path, 'data', 'SherlockHolmes.txt'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    textInput = request.args.get("text")
    maxNumberSuggestions = 15
    suggestions = nextWordPrediction(textInput, modelVocab, model, maxNumberSuggestions)
    return jsonify(suggestions)

if __name__=='__main__': 
   app.run(debug=True) 