import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
from collections import Counter

ps = PorterStemmer()

class AccuracyTesting:

    def predict(self, heuristics, heuristic, message, correctScore, threshold1, threshold2, threshold_common):
        if heuristic in heuristics:
            heuDict = heuristics[heuristic]
        else:
            print('This heuristic had too low of a sample size: ', heuristic)
            return False
        words = word_tokenize(message)
        stemmed_words = []
        for w in words:
            stemmed_words.append(ps.stem(w))
        commonCount = 0
        for w in stemmed_words:
            if w in heuDict:
                if heuDict[w] >= threshold_common:
                    commonCount += 1
        if commonCount >= threshold1:
            answer = 1
        elif commonCount >= threshold2:
            answer = 2
        else:
            answer = 3
        if answer == correctScore:
            return True
        else:
            return False

    def runAccuracyTest(self, heuristics, testing_data, inp1, inp2, inp3):
        numRight = 0
        for i in range(0, testing_data.shape[0]):
            heuristic = testing_data['HeuristicName'][i]
            message = testing_data['Messages'][i]
            correctScore = testing_data['Manual Heuristic Score'][i]
            if(self.predict(heuristics, heuristic, message, correctScore, inp1, inp2, inp3) == True):
                numRight+=1

        answerPercent = (numRight/testing_data.shape[0]) * 100
        results = [answerPercent, inp1, inp2, inp3]
        return results