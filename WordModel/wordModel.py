import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from WordModel.wordTraining import PrepTrainingData
from WordModel.wordTesting import PrepTestingData
ps = PorterStemmer()

class WordModel:

    def __init__(self, trainingFile, testingFile, wordOutputFile):
        self.trainingFile = trainingFile
        #print(testingFile)
        self.testingFile = testingFile
        self.wordOutputFile = wordOutputFile
    
    def loadTrainingData(self, loadRaw=False, threshold=5):
        self.training = PrepTrainingData(self.trainingFile)
        if(loadRaw==True):
            self.training.getRawData()
            self.training.dropOrigAndLow()
        else:
            self.training.getReadyData()
        self.training.sortSentiments()
        self.training.sortWords()
        self.training.cleanLowVariety(threshold)
        self.training.removeStopwords()
        self.training.cleanLowSampleSizes(2)
        self.training.toCsv(self.wordOutputFile)
        self.data = self.training.getData()
        #print(data.shape[0])
        self.sentiments = self.training.getSentiments()
        self.sentimentCounts = self.training.getSentimentCounts()
        #print("Model has ", len(sentiments), "")

    def loadTestingData(self):
        self.testing = PrepTestingData(self.testingFile)
        self.testing.dropOrig()
        self.testing_data = self.testing.getTestingData()

    def getTrainingData(self):
        return self.data

    def getTestingData(self):
        return self.testing_data

    def getSentiments(self):
        return self.sentiments