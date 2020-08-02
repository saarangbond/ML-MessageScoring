import progressbar as pb
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import words, stopwords
import csv

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class PrepTrainingData:
    def __init__(self, trainingFile):
        self.sentiments = {}
        self.sentimentCounts = {}
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
            if not sent in self.sentiments:
                self.sentiments[sent] = {}
        print('SORTED ', len(self.sentiments), ' SENTIMENTS')

    def sortWords(self):
        count = 0
        for i in range(0,self.data.shape[0]):
            message = str(self.data['Messages'][i])
            message_tokens = word_tokenize(message)
            sentimentName = str(self.data['SentimentName'][i])
            sentimentName = sentimentName.upper()
            for tok in message_tokens:
                if isinstance(tok, str):
                    root = lemmatizer.lemmatize(str(tok))
                    root = root.lower()
                    if not root in self.sentiments[sentimentName]:
                        self.sentiments[sentimentName][root] = 1
                        count += 1
                    else:
                        self.sentiments[sentimentName][root] += 1
                
        print('PROCESSED ', count, ' MESSAGE WORDS')

    def removeStopwords(self):
        count = 0
        sCount = 0
        initial_stopwords = set(stopwords.words('English'))
        stop_words = []
        for w in initial_stopwords:
            stop_words.append((lemmatizer.lemmatize(w)).lower())
        custom_sw = ( 'we', '$', '%', '.', '*', ',', '!', ':', '?', ';', '(', ')', '\\', '\'s', 'a', 'in', 'patient', 'product', 'treatment', 'if', 'dont', 'I', 'n\'t', '-', 'help', 'bosulif', 'do', 'go365', 'pfizer', '\\xe2\\x80\\x99')       
        for w in custom_sw:
            stop_words.append(w.lower())
        sKeys = self.sentiments.keys()
        
        progress = pb.ProgressBar(max_value=len(sKeys))
        progvar = 0
        for s in sKeys:
            sName = str(s)
            sName = sName.upper()
            delList = []
            for w in self.sentiments[sName]:
                if w in delList:
                    continue
                if w in stop_words:
                    delList.append(w)
                    count += 1
                    continue
                if w not in words.words():
                    delList.append(w)
                    count += 1
            for d in delList:
                if d in self.sentiments[sName]:
                    del self.sentiments[sName][d]
        if len(self.sentiments[sName]) == 0:
            del self.sentiments[sName]
            sCount += 1
            progvar += 1
            progress.update(progvar)
        print('\nREMOVED ', count, ' JUNK WORDS AND ', sCount, ' SENTIMENTS')

    def cleanLowSampleSizes(self, threshold):
        delete = []
        counter = 0
        for sent in self.sentimentCounts.keys():
            if self.sentimentCounts[sent] < threshold:
                delete.append(sent)
                if sent in self.sentiments:
                    del self.sentiments[sent]
                counter+=1
        for key in delete:
            del self.sentimentCounts[key]            
        print('CLEANED LOW SAMPLE SIZES', counter)
        self.data.reset_index(drop=True, inplace = True)
    
    def cleanLowVariety(self, threshold):
        befLength = len(self.sentiments.keys())
        delete = []
        counter = 0
        for sent in self.sentiments.keys():
            print((len(self.sentiments[sent].keys())))
            if len(self.sentiments[sent].keys()) < threshold:
                delete.append(sent)
                counter+=1
        for key in delete:
            del self.sentiments[key]
        aftLength = len(self.sentiments.keys())
        print('CLEANED LOW VARIETY', counter)
        print('STARTED WITH ', befLength, " SENTIMENTS, ENDED WITH ", aftLength)
        self.data.reset_index(drop=True, inplace = True)

    def toCsv(self, fName):
        w = csv.writer(open(fName, 'w'))
        for sName in self.sentiments:
            sDict = self.sentiments[sName]
            w.writerow([sName, ':'])
            for word in sorted(sDict, key=sDict.get, reverse=True):
                #print(str(k), v)
                val = sDict[word]
                sWord = str(word).encode('utf-8')
                w.writerow([sWord, val])
            #w.writerow(['', ''])

    def getData(self):
        return self.data
    
    def getSentimentCounts(self):
        return self.sentimentCounts
    
    def getSentiments(self):
        return self.sentiments
    