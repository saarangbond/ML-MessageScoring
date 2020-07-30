import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import stopwords
import itertools, sys
import csv

bigram_measures = BigramAssocMeasures()
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class PrepTrainingData:
    def __init__(self):
        self.heuristicCounts = {}
        self.heuristicPhrases = {}
    
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
            if not heu in self.heuristicPhrases:
                self.heuristicPhrases[heu] = {}
        print('SORTED HEURISTICS')

    def sortPhrases(self):
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        for i in range(0,self.data.shape[0]):
            sys.stdout.write(next(spinner))
            sys.stdout.flush()
            message = str(self.data['Messages'][i])
            message = message.lower()
            tokens = []
            punc = ['.', ',', '!', '?', ';', ':', '\'', '--', '\'nt', '\'ll', '{', '}', '\\', '(', ')', 'product x']
            for w in word_tokenize(message):
                if w not in stopwords.words('english') and w not in punc:
                    tokens.append(w)
            heuristicName = str(self.data['HeuristicName'][i])
            heuristicName = heuristicName.upper()
            finder = BigramCollocationFinder.from_words(tokens)
            msgPhrases = finder.nbest(bigram_measures.pmi, 6)
            for m in msgPhrases:
                ph = str(m[0]) + " " + str(m[1])
                if isinstance(ph, str):
                    if ph not in self.heuristicPhrases[heuristicName]:
                        self.heuristicPhrases[heuristicName][ph] = 1
                    elif ph in self.heuristicPhrases[heuristicName]:
                        self.heuristicPhrases[heuristicName][ph] += 1
            sys.stdout.write('\b')
            
            
        print('PROCESSED ALL MESSAGE PHRASES')

    def cleanLowSampleSizes(self, threshold):
        delete = []
        counter = 0
        for heu in self.heuristicCounts.keys():
            if self.heuristicCounts[heu] < threshold:
                delete.append(heu)
                del self.heuristicPhrases[heu]
                counter+=1
        for key in delete:
            del self.heuristicCounts[key]
        print('CLEANED LOW SAMPLE SIZES', counter)
        self.data.reset_index(drop=True, inplace = True)
    
    def cleanLowVariety(self, threshold):
        delete = []
        counter = 0
        for heu in self.heuristicPhrases.keys():
            if len(self.heuristicPhrases[heu]) < threshold:
                delete.append(heu)
                counter+=1
        for key in delete:
            del self.heuristicPhrases[key]
        print('CLEANED LOW VARIETY', counter)
        self.data.reset_index(drop=True, inplace = True)

    def toCsv(self):
        w = csv.writer(open('phraseDict.csv', 'w'))
        for key in self.heuristicPhrases:
            w.writerow([key, ':'])
            for k, v in self.heuristicPhrases[key].items():
                w.writerow(([str(k).encode('utf-8'), str(v).encode('utf-8')]))
            w.writerow(['', ''])

    def getData(self):
        return self.data
    
    def getHeuristicCounts(self):
        return self.heuristicCounts
    
    def getHeuristicPhrases(self):
        return self.heuristicPhrases

    