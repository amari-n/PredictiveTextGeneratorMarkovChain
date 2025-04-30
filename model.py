# -*- coding: utf-8 -*-
import nltk
import string
import numpy
import collections
import random

def loadModel(txtPath):
    # Open the SherlockHolmes file and read the text
    words = tokenize(txtPath)

    n_gramSize = 2
    n_grams = generateN_grams(n_gramSize, words)

    unique = {word: i for i, word in enumerate(sorted(set(words)))}
    transitionMatrix = buildTransitionMatrix(words, n_grams)

    return unique, transitionMatrix

# Takes txt as input and returns a list of the words within the text without punctuation and all lowercase
def tokenize(txt):
    # Tokenize (aka preprocess) the data
    text = open(txt, "r", encoding="utf-8")
    punctuation = '\ufeff“”‘’«»—…™' + string.punctuation
    translator = str.maketrans('', '', punctuation)
    singleLetters = ['a', 'i']

    words = []
    for line in text:
        line = line.replace("-", " ")
        line = line.translate(translator).lower()
        tempWords = line.split()
        wordsToAdd = [w for w in tempWords if w != '' and (len(w) > 1 or w in singleLetters)]
        words += wordsToAdd
    return words

# Takes n_gram size and text word list as input and returns n_grams
def generateN_grams(n, words):
    #Generate n-grams to use for Transition Matrix
    n_grams = nltk.ngrams(words, n)
    n_grams = list(n_grams)
    return n_grams

# Takes text word list and n_grams list as input and builds transitionMatrix with word probabilities
def buildTransitionMatrix(words, n_grams):
    #Build Transition Matrix
    unique = {word: i for i, word in enumerate(sorted(set(words)))}
    transitionMatrix = numpy.zeros((len(unique), len(unique)))

    #Populate Transition Matrix
    for (word1, word2) in n_grams:
        transitionMatrix[unique[word1]][unique[word2]] += 1

    #Calculate Probabilities
    rowCount = transitionMatrix.sum(axis=1, keepdims=True)
    rowCount[rowCount == 0] = 1
    transitionMatrix /= rowCount
    return transitionMatrix

def nextWordPrediction(sampleSentence, unique, transitionMatrix, n):
    lastWord = sampleSentence.lower().strip().split()[-1]
    
    if lastWord not in unique:
        print("Word not in vocabulary.")
        return []
    
    probabilities = transitionMatrix[unique[lastWord]]
    probabilityDict = {}
    wordIdxCounter = 0
    for p in probabilities:
        if p in probabilityDict.keys():
            probabilityDict[p].append(wordIdxCounter)
        elif p != 0:
            probabilityDict[p] = [wordIdxCounter]
        wordIdxCounter += 1
    probabilityDict = collections.OrderedDict(sorted(probabilityDict.items()))
    
    suggestions = []
    suggestionsCount = n
    probabilityDictIdx = len(probabilityDict.keys())-1
    while suggestionsCount > 0 and probabilityDictIdx >= 0:
        p = list(probabilityDict.keys())[probabilityDictIdx]
        if (len(probabilityDict[p]) > suggestionsCount):
            suggestions += random.sample(probabilityDict[p], suggestionsCount)
            suggestionsCount -= len(probabilityDict[p])
        else:
            suggestions += probabilityDict[p]
            suggestionsCount -= len(probabilityDict[p])
            probabilityDictIdx -= 1
    
    wordSuggestions = []
    for idx in suggestions:
        wordSuggestions.append(list(unique.keys())[idx])
    return wordSuggestions