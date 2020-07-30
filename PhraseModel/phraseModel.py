import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from PhraseModel.loadTraining import PrepTrainingData
from PhraseModel.loadTesting import PrepTestingData
ps = PorterStemmer()

class PhraseModel:
    
    def loadTrainingData(self, loadRaw=False):
        self.training = PrepTrainingData()
        if(loadRaw==True):
            self.training.getRawData()
            self.training.dropOrigAndLow()
        else:
            self.training.getReadyData()
        self.training.sortHeuristics()
        self.training.sortPhrases()
        self.training.cleanLowSampleSizes(2)
        self.training.cleanLowVariety(6)
        
        self.data = self.training.getData()
        #print(data.shape[0])
        self.heuristicPhrases = self.training.getHeuristicPhrases()
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

    def getHeuristicPhrases(self):
        return self.heuristicPhrases