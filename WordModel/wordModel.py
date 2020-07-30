import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from WordModel.loadTraining import PrepTrainingData
from WordModel.loadTesting import PrepTestingData
ps = PorterStemmer()

class WordModel:
    
    def loadTrainingData(self, loadRaw=False):
        self.training = PrepTrainingData()
        if(loadRaw==True):
            self.training.getRawData()
            self.training.dropOrigAndLow()
        else:
            self.training.getReadyData()
        self.training.sortHeuristics()
        self.training.sortWords()
        self.training.removeStopwords()
        self.training.cleanLowSampleSizes(2)
        self.training.cleanLowVariety(6)
        self.training.toCsv()
        self.data = self.training.getData()
        #print(data.shape[0])
        self.heuristics = self.training.getHeuristics()
        self.heuristicCounts = self.training.getHeuristicCounts()
        #print("Model has ", len(heuristics), "")

    def loadTestingData(self, testFile):
        self.testing = PrepTestingData(testFile)
        self.testing.dropOrig()
        self.testing_data = self.testing.getTestingData()

    def getTrainingData(self):
        return self.data

    def getTestingData(self):
        return self.testing_data

    def getHeuristics(self):
        return self.heuristics