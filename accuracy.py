#pylint: disable-all
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class AccuracyTesting:

    def __init__(self):
        self.lowSampleSizeHeuristics = []

    def predict(self, heuristics, heuristic, message, correctScore):
        if heuristic in heuristics:
            heuDict = heuristics[heuristic]
        elif heuristic not in heuristics:
            if heuristic not in self.lowSampleSizeHeuristics:
                self.lowSampleSizeHeuristics.append(heuristic)
            return False
        words = word_tokenize(message)
        root_words = []
        for w in words:
            root_words.append(lemmatizer.lemmatize(w))

        c = Counter(heuDict)
        common = c.most_common(5)
        present1 = False
        present2 = False
        present3 = False
        present4 = False
        present5 = False
        for w in root_words:
            if w == (common[0])[0]:
                present1 = True
            elif w == (common[1])[0]:
                present2 = True
            elif w == (common[2])[0]:
                present3 = True
            elif w == (common[3])[0]:
                present4 = True
            elif w == (common[4])[0]:
                present5 = True
        total = 0
        denom = (common[0])[1] + (common[1])[1] + (common[2])[1] + (common[3])[1] + (common[4])[1]
        weight1 = ((common[0])[1]) / denom
        weight2 = ((common[1])[1]) / denom
        weight3 = ((common[2])[1]) / denom
        weight4 = ((common[3])[1]) / denom
        weight5 = ((common[4])[1]) / denom
        if present1 == True:
            total += weight1
        if present2 == True:
            total += weight2
        if present3 == True:
            total += weight3
        if present4 == True:
            total += weight4
        if present5 == True:
            total += weight5
        if total >= (weight4):
            answer = 1
        elif total >= (weight5):
            answer = 2
        elif total < (weight5):
            answer = 3
        if answer == correctScore:
            return True
        else:
            return False

    def runAccuracyTest(self, heuristics, testing_data):
        numRight = 0
        for i in range(0, testing_data.shape[0]):
            heuristic = (testing_data['HeuristicName'][i]).upper()
            message = testing_data['Messages'][i]
            correctScore = testing_data['Manual Heuristic Score'][i]
            if heuristic not in heuristics:
               print("Did not find ", heuristic, " in trained model with ", len(heuristics), " heuristics")
               self.addHeuristic(heuristic, heuristics, message)
               continue
            if(self.predict(heuristics, heuristic, message, correctScore) == True):
                numRight+=1
            self.updateHeuristicWords(heuristic, heuristics, message)

        answerPercent = (numRight/testing_data.shape[0]) * 100
        results = [answerPercent]
        print(results)
        return results

    def updateHeuristicWords(self, heuristic, heuristics, message):
        if heuristic in heuristics:
            initial_stopwords = set(stopwords.words('English'))
            stop_words = []
            for w in initial_stopwords:
                stop_words.append(lemmatizer.lemmatize(w))
            custom_sw = ( 'we', '$', '%', '.', '*', ',', '!', ':', '?', ';', '(', ')', '\'s', 'a', 'in', 'patient', 'product', 'treatment', 'if', 'dont', 'I', 'n\'t', '-', 'help', 'bosulif', 'do', 'go365', 'pfizer')
            message_tokens = word_tokenize(message)
            for tok in message_tokens:
                root = lemmatizer.lemmatize(str(tok))
                root = root.lower()
                if not root in stop_words and not root in custom_sw:
                    if not root in heuristics[heuristic]:
                        (heuristics[heuristic])[root] = 1
                    else:
                        (heuristics[heuristic])[root] += 1
    
    def addHeuristic(self, heuristic, heuristics, message):
        if heuristic not in heuristics:
            heuristics[heuristic] = {}
            initial_stopwords = set(stopwords.words('English'))
            stop_words = []
            for w in initial_stopwords:
                stop_words.append(lemmatizer.lemmatize(w))
            custom_sw = ( 'We', '$', '%', '.', '*', ',', '!', ':', '?', ';', '(', ')', '\'s', 'A', 'In', 'patient', 'product', 'treatment', 'If', 'dont', 'I', 'n\'t', '-', 'help', 'bosulif', 'Do', 'go365', 'pfizer')
            message_tokens = word_tokenize(message)
            for tok in message_tokens:
                root = lemmatizer.lemmatize(str(tok))
                if not root in stop_words and not root in custom_sw:
                    if not root in heuristics[heuristic]:
                        (heuristics[heuristic])[root] = 1
                    else:
                        (heuristics[heuristic])[root] += 1
    
