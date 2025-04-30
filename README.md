# Predictive Text Generator Using Markov Chain

This project is a Natural Language Processing (NLP) web application that predicts the next word in the sentence using a Markov Chain. The model is trained using Sherlock Holmes' writing patterns.

## Web App Description and DEMO

Using the input corpus (`SherlockHolmes.txt`) the model is trained and will generate predictive text suggestions for the next most probable word in a sentence of the user's choice typed in the chatbox. Suggestions appear in the form of buttons labeled with the word suggestions. When pressed, the buttons will add the respective word to the sentence, and more suggestions will be available again.

Web Application: [DEMO VIDEO LINK](https://www.dropbox.com/scl/fi/3xoh0mel8p89i2qkk9ho2/Predictive-Text-Generator-Google-Chrome-2025-04-30-19-30-54.mp4?rlkey=j6p9ckh8xd6r89ieagm9mp1rt&st=lyvy312l&dl=0)

## How The Model Works `model.py`

1. The training data is preprocessed in the `def tokenize(txt)` function, meaning all punctuation is removed from the text file and it is made lowercase.
2. Using the preprocessed data, n grams of size 2 (bigrams) are created using `generateN_grams(n, words)` to be used in the transition matrix in order to calculate word probabilities.
3. Now, using both the preprocessed data and our list of n_grams, the transition probability matrix of our Markov chain is created using `def buildTransitionMatrix(words, n_grams)`. This function calculates the probabilities necessary in order to find the most probable suggested word.
4. Lastly, `def nextWordPrediction(sampleSentence, unique, transitionMatrix, n)` predicts the top `n` most probable next words based on user input. This is done by analyzing the Markov transition probabilities based on the last word in the user's input sentence.

## How To Run The Application Locally

* Clone the repository <br>`git clone https://github.com/amari-n/PredictiveTextGeneratorMarkovChain.git` <br> `cd PredictiveTextGeneratorMarkovChain`

* Create and activate a virtual environment <br> `python -m venv venv` <br>Windows: `.\venv\Scripts\activate`

* Install dependencies `pip install -r requirements.txt`

* Run the application `python -m flask run`

* Open [http://localhost:5000/](http://localhost:5000/) in your browser.

## Languages and Tools Used

* `Python` - Backend Logic
* `NLTK` - Text Processing
* `Markov Chain` - Prediction Algorithm
* `Flask` - Web Framework
* `HTML/CSS/JavaScript` - Frontend Logic

## Author
Nizar Amari
