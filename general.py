import getopt
import os, sys
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from WordModel.wordModel import WordModel
from PhraseModel.phraseModel import PhraseModel
from overall import OverallScore
import unicodedata
import argparse

parser = argparse.ArgumentParser(description='Process cmdline args')

parser.add_argument('--data', dest='data', choices=['processed', 'raw'], type=str)
parser.add_argument('--test', dest='test', choices=[True, False], type=bool)
parser.add_argument('--trainingFile', dest='trainingFile', type=str)
parser.add_argument('--testingFile', dest='testingFile', type=str)
parser.add_argument('--wordFile', dest='wordFile', type=str)
parser.add_argument('--phraseFile', dest='phraseFile', type=str)
parser.add_argument('--doWordModel', dest='doWordModel', type=bool)
parser.add_argument('--doPhraseModel', dest='doPhraseModel', type=bool)
parser.add_argument('--sentimentWordThreshold', dest='sentimentWordThreshold', type=int)

cwd = os.getcwd()
#print(cwd)
loadRaw = False
test = False
data = 'processed'
trainingFile = 'SampleTrainingData.xlsx'
testingFile = 'SampleTestingData.xlsx'
wordFile = 'wordDict.csv'
phraseFile = 'phraseDict.csv'
doWordModel = True
doPhraseModel = True
sentimentWordThreshold = 5

args = parser.parse_args()
#print(args)

if 'data' in args:
    data = getattr(args, 'data')
    if data == 'processed':
        loadRaw = False
    elif data == 'raw':
        loadRaw = True
if 'test' in args:
    if getattr(args, 'test') != None:
        test = getattr(args, 'test')
if 'trainingFile' in args:
    if getattr(args, 'trainingFile') != None:
        trainingFile = getattr(args, 'trainingFile')
if 'testingFile' in args:
    if getattr(args, 'testingFile') != None:
        testingFile = getattr(args, 'testingFile')
if 'wordFile' in args:
    if getattr(args, 'wordFile') != None:
        wordFile = getattr(args, 'wordFile')
if 'phraseFile' in args:
    if getattr(args, 'phraseFile') != None:
        phraseFile = getattr(args, 'phraseFile')
if 'doWordModel' in args:
    if getattr(args, 'doWordModel') != None:
        doWordModel = getattr(args, 'doWordModel')
if 'doPhraseModel' in args:
    if getattr(args, 'doPhraseModel') != None:
        doPhraseModel = getattr(args, 'doPhraseModel')
if 'sentimentWordThreshold' in args:
    if getattr(args, 'sentimentWordThreshold') != None:
        sentimentWordThreshold = getattr(args, 'sentimentWordThreshold')

trainingPath = cwd + '\\' + trainingFile
testingPath = cwd + '\\' + testingFile

ps = PorterStemmer()
wordModel = WordModel(trainingPath, testingPath, wordFile)
phraseModel = PhraseModel(trainingPath, testingPath, phraseFile)

print('\nAutomatically Score Marketing Messages')
    
if doWordModel == True:
    wordModel.loadTrainingData(loadRaw, sentimentWordThreshold)
    wordModel.loadTestingData()
    sentiments = wordModel.getSentiments()
    print('WORD MODEL PREP COMPLETE\n')

if doPhraseModel == True:
    phraseModel.loadTrainingData(loadRaw)
    phraseModel.loadTestingData()
    sentimentPhrases = phraseModel.getSentimentPhrases()
    print('PHRASE MODEL PREP COMPLETE\n')

if test == False:
    exit()

testing_data = wordModel.getTestingData()
overallScore = OverallScore(testing_data)
bestTesting = [0, 0, 0]
for x in range(50,51):
    y = 100-x
    testing = overallScore.runScores(sentimentPhrases, sentiments, x, y, doWordModel, doPhraseModel)
    if(testing[0] > bestTesting[0]):
        bestTesting = testing
print('Best: ', bestTesting)



