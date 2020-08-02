import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from PhraseModel.phraseTraining import PrepTrainingData
from PhraseModel.phraseTesting import PrepTestingData
ps = PorterStemmer()

class PhraseModel:
    
    def __init__(self, trainingFile, testingFile, phraseOutputFile):
        self.trainingFile = trainingFile
        self.testingFile = testingFile
        self.phraseOutputFile = phraseOutputFile

    def loadTrainingData(self, loadRaw=False):
        self.training = PrepTrainingData(self.trainingFile)
        if(loadRaw==True):
            self.training.getRawData()
            self.training.dropOrigAndLow()
        else:
            self.training.getReadyData()
        self.training.sortSentiments()
        self.training.sortPhrases()
        self.training.cleanLowSampleSizes(2)
        self.training.cleanLowVariety(6)
        self.training.toCsv(self.phraseOutputFile)
        
        self.data = self.training.getData()
        #print(data.shape[0])
        self.sentimentPhrases = self.training.getSentimentPhrases()
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

    def getSentimentPhrases(self):
        return self.sentimentPhrases