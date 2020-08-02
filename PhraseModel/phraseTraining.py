import progressbar as pb
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import words, stopwords
import itertools, sys
import csv

bigram_measures = BigramAssocMeasures()
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class PrepTrainingData:
    def __init__(self, trainingFile):
        self.sentimentCounts = {}
        self.sentimentPhrases = {}
        self.trainingFile = trainingFile
    
    def getReadyData(self):
        self.data = pd.read_excel(self.trainingFile)
        print('LOADED DATA')

    def getRawData(self):
        self.data = pd.read_excel(self.trainingFile)
        self.readyData = pd.DataFrame(columns=["SentimentName","Messages","ManualScore","MessageType"])
        for i in range(0,self.data.shape[0]):
            print(i)
            sentiment = str(self.data['SentimentName'][i])
            message = str(self.data['Messages'][i])
            score = str(self.data['ManualScore'][i])
            messageType = str(self.data['MessageType'][i])
            self.readyData = self.readyData.append({'SentimentName': sentiment, 'Messages':message, 'ManualScore': score, 'MessageType': messageType}, ignore_index=True)

        print('LOADED DATA')

    def dropOrigAndLow(self):
        self.data = self.data.loc[self.data['MessageType'] != 'Original Message']
        print('DROPPED ORIGINAL MESSAGES')
        self.data = self.data.loc[self.data['ManualScore'] == 1.0]
        print('DROPPED 2 AND 3 SCORES')
        self.data.reset_index(drop=True, inplace=True)
        self.readyData.to_excel('Data.xlsx')

    def sortSentiments(self):
        for i in range(0,self.data.shape[0]):
            sent = str(self.data['SentimentName'][i])
            sent = sent.upper()
            if sent not in self.sentimentCounts:
                self.sentimentCounts[sent] = 1
            else:
                self.sentimentCounts[sent] += 1
            if not sent in self.sentimentPhrases:
                self.sentimentPhrases[sent] = {}
        print('SORTED SENTIMENTS')

    def sortPhrases(self):
        progress = pb.ProgressBar(max_value=self.data.shape[0])
        for i in range(0,self.data.shape[0]):
            message = str(self.data['Messages'][i])
            message = message.lower()
            tokens = []
            punc = ['.', ',', '!', '?', ';', ':', '\'', '--', '\'nt', '\'ll', '{', '}', '\\', '(', ')', 'product x']
            for w in word_tokenize(message):
                if w not in stopwords.words('english') and w not in punc and w in words.words():
                    tokens.append(w)
            sentimentName = str(self.data['SentimentName'][i])
            sentimentName = sentimentName.upper()
            finder = BigramCollocationFinder.from_words(tokens)
            msgPhrases = finder.nbest(bigram_measures.pmi, 6)
            for m in msgPhrases:
                ph = str(m[0]) + " " + str(m[1])
                if isinstance(ph, str):
                    if ph not in self.sentimentPhrases[sentimentName]:
                        self.sentimentPhrases[sentimentName][ph] = 1
                    elif ph in self.sentimentPhrases[sentimentName]:
                        self.sentimentPhrases[sentimentName][ph] += 1
            progress.update(i)
        print('PROCESSED ALL MESSAGE PHRASES AND REMOVED STOPWORDS')

    def cleanLowSampleSizes(self, threshold):
        delete = []
        counter = 0
        for sent in self.sentimentCounts.keys():
            if self.sentimentCounts[sent] < threshold:
                delete.append(sent)
                del self.sentimentPhrases[sent]
                counter+=1
        for key in delete:
            del self.sentimentCounts[key]
        print('CLEANED LOW SAMPLE SIZES', counter)
        self.data.reset_index(drop=True, inplace = True)
    
    def cleanLowVariety(self, threshold):
        delete = []
        counter = 0
        for sent in self.sentimentPhrases.keys():
            if len(self.sentimentPhrases[sent]) < threshold:
                delete.append(sent)
                counter+=1
        for key in delete:
            del self.sentimentPhrases[key]
        print('CLEANED LOW VARIETY', counter)
        self.data.reset_index(drop=True, inplace = True)

    def toCsv(self, file):
        w = csv.writer(open(file, 'w'))
        for key in self.sentimentPhrases:
            w.writerow([key, ':'])
            for k, v in self.sentimentPhrases[key].items():
                w.writerow(([str(k).encode('utf-8'), str(v).encode('utf-8')]))
            w.writerow(['', ''])

    def getData(self):
        return self.data
    
    def getSentimentCounts(self):
        return self.sentimentCounts
    
    def getSentimentPhrases(self):
        return self.sentimentPhrases

    