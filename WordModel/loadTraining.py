import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import csv

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class PrepTrainingData:
    def __init__(self):
        self.heuristics = {}
        self.heuristicCounts = {}
    
    def getReadyData(self):
        self.data = pd.read_excel(r'C:\Users\Saarang\Documents\NewristicsScoringModel\Data.xlsx')
        print('LOADED DATA')

    def getRawData(self):
        self.data = pd.read_excel(r'C:\Users\Saarang\Documents\NewristicsScoringModel\MessageData.xlsx')
        self.readyData = pd.DataFrame(columns=["HeuristicName","Messages","ManualHeuristicScore","MessageType"])
        for i in range(0,self.data.shape[0]):
            print(i)
            heuristic = str(self.data['HeuristicName'][i])
            message = str(self.data['Messages'][i])
            score = str(self.data['ManualHeuristicScore'][i])
            messageType = str(self.data['MessageType'][i])
            self.readyData = self.readyData.append({'HeuristicName': heuristic, 'Messages':message, 'ManualHeuristicScore': score, 'MessageType': messageType}, ignore_index=True)
        print('LOADED DATA')

    def dropOrigAndLow(self):
        self.data = self.data.loc[self.data['MessageType'] != 'Original Message']
        print('DROPPED ORIGINAL MESSAGES')
        self.data = self.data.loc[self.data['ManualHeuristicScore'] == 1.0]
        print('DROPPED 2 AND 3 SCORES')
        self.data.reset_index(drop=True, inplace=True)
        self.readyData.to_excel('Data.xlsx')

    def sortHeuristics(self):
        for i in range(0,self.data.shape[0]):
            heu = str(self.data['HeuristicName'][i])
            heu = heu.upper()
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
            heuristicName = heuristicName.upper()
            for tok in message_tokens:
                if isinstance(tok, str):
                    root = lemmatizer.lemmatize(str(tok))
                    root = root.lower()
                    if not root in self.heuristics[heuristicName]:
                        self.heuristics[heuristicName][root] = 1
                    else:
                        self.heuristics[heuristicName][root] += 1
                
        print('PROCESSED ALL MESSAGE WORDS')

    def removeStopwords(self):
        initial_stopwords = set(stopwords.words('English'))
        stop_words = []
        for w in initial_stopwords:
            stop_words.append((lemmatizer.lemmatize(w)).lower())
        custom_sw = ( 'we', '$', '%', '.', '*', ',', '!', ':', '?', ';', '(', ')', '\\', '\'s', 'a', 'in', 'patient', 'product', 'treatment', 'if', 'dont', 'I', 'n\'t', '-', 'help', 'bosulif', 'do', 'go365', 'pfizer')       
        for i in range(0,self.data.shape[0]):
            heuristicName = str(self.data['HeuristicName'][i])
            heuristicName = heuristicName.upper()
            for w in stop_words:
                if w in self.heuristics[heuristicName]:
                    del self.heuristics[heuristicName][w]
            for c in custom_sw:
                if c in self.heuristics[heuristicName]:
                    del self.heuristics[heuristicName][c]
        print('REMOVED JUNK WORDS')

    def cleanLowSampleSizes(self, threshold):
        delete = []
        counter = 0
        for heu in self.heuristicCounts.keys():
            if self.heuristicCounts[heu] < threshold:
                delete.append(heu)
                del self.heuristics[heu]
                counter+=1
        for key in delete:
            del self.heuristicCounts[key]
        print('CLEANED LOW SAMPLE SIZES', counter)
        self.data.reset_index(drop=True, inplace = True)
    
    def cleanLowVariety(self, threshold):
        delete = []
        counter = 0
        for heu in self.heuristics.keys():
            if len(self.heuristics[heu].keys()) < threshold:
                delete.append(heu)
                counter+=1
        for key in delete:
            del self.heuristics[key]
        print('CLEANED LOW VARIETY', counter)
        self.data.reset_index(drop=True, inplace = True)

    def toCsv(self):
        w = csv.writer(open('wordDict.csv', 'w'))
        for key in self.heuristics:
            w.writerow([key, ':'])
            for k, v in self.heuristics[key].items():
                #print(str(k), v)
                w.writerow(([str(k).encode('utf-8'), str(v).encode('utf-8')]))
            w.writerow([' ', ' '])

    def getData(self):
        return self.data
    
    def getHeuristicCounts(self):
        return self.heuristicCounts
    
    def getHeuristics(self):
        return self.heuristics
    