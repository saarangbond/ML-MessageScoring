import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()

class PrepTrainingData:
    def __init__(self):
        self.data = pd.read_excel(r'C:\Users\Saarang\Documents\NewristicsScoringModel\MessageData.xlsx')
        self.heuristics = {}
        self.heuristicCounts = {}
        print('LOADED DATA')
    
    def dropOrigAndLow(self):
        self.data = self.data.loc[self.data['MessageType'] != 'Original Message']
        print('DROPPED ORIGINAL MESSAGES')
        self.data = self.data.loc[self.data['ManualHeuristicScore'] == 1.0]
        print('DROPPED 2 AND 3 SCORES')
        self.data.reset_index(drop=True, inplace=True)

    def sortHeuristics(self):
        for i in range(0,self.data.shape[0]):
            heu = str(self.data['HeuristicName'][i])
            if heu not in self.heuristicCounts:
                self.heuristicCounts[heu] = 1
            else:
                self.heuristicCounts[heu] += 1
            if not heu in self.heuristics:
                self.heuristics[heu] = {}
        print('SORTED HEURISTICS')

    def sortWords(self):
        for i in range(0,self.data.shape[0]):
            message = str(self.data['Messages'][i])
            message_tokens = word_tokenize(message)
            heuristicName = str(self.data['HeuristicName'][i])
            for tok in message_tokens:
                stem = ps.stem(str(tok))
                if not stem in self.heuristics[heuristicName]:
                    self.heuristics[heuristicName][stem] = 1
                else:
                    self.heuristics[heuristicName][stem] += 1
        print('PROCESSED ALL MESSAGE WORDS')

    def removeStopwords(self):
        stop_words = set(stopwords.words('English'))
        custom_sw = ( '.', ',', '!', ':', '?', ';', '(', ')', '\'s', 'A', 'In', 'patient', 'treatment', 'If', 'dont', 'I', 'n\'t', '-', 'help', 'bosulif', 'Do', 'go365', 'pfizer')
        for i in range(0,self.data.shape[0]):
            heuristicName = str(self.data['HeuristicName'][i])
            for w in stop_words:
                if w in self.heuristics[heuristicName]:
                    del self.heuristics[heuristicName][w]
            for c in custom_sw:
                if c in self.heuristics[heuristicName]:
                    del self.heuristics[heuristicName][c]
        print('REMOVED JUNK WORDS')

    def cleanLowSampleSizes(self, threshold):
        for heu in self.heuristicCounts.keys():
            if self.heuristicCounts[heu] < threshold:
                del self.heuristicCounts[heu]
                del self.heuristics[heu]

    def cleanLowVariety(self, threshold):
        for heu in self.heuristics:
            if self.heuristics[heu].keys().length < threshold:
                del self.heuristics[heu]

    def getData(self):
        return self.data
        
    def getHeuristics(self):
        return self.heuristics
    