import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from WordModel.accuracy import WordAccuracyTesting
from PhraseModel.accuracy import PhraseAccuracyTesting
import statistics
ps = PorterStemmer()

class OverallScore:

    def __init__(self, testing_data):
        self.outputDf = pd.DataFrame(columns=["Heuristic","Message","Actual Score","Predicted Overall Score"])
        self.phraseDf = pd.DataFrame(columns=["Heuristic","Message","Actual Score","Predicted Phrase Score"])
        self.wordDf = pd.DataFrame(columns=["Heuristic","Message","Actual Score","Predicted Word Score"])
        self.phraseAccuracy = PhraseAccuracyTesting()
        self.wordAccuracy = WordAccuracyTesting()
        self.testing_data = testing_data

    def runScores(self, heuristicPhrases, heuristics, x, y):
        numRight = 0
        totalCases = self.testing_data.shape[0]
        for i in range(0,self.testing_data.shape[0]):
            heuristic = self.testing_data['HeuristicName'][i]
            heuristic = heuristic.upper()
            message = self.testing_data['Messages'][i]
            message = message.lower()
            correctScore = self.testing_data['Manual Heuristic Score'][i]
            wordResults = self.wordAccuracy.runAccuracyTest(heuristics, heuristic, message, correctScore)
            phraseResults = self.phraseAccuracy.runAccuracyTest(heuristicPhrases, heuristic, message, correctScore)
            score = self.getOverallScore(wordResults, phraseResults, x, y)
            if wordResults[2] == True and phraseResults[2] == True:
                totalCases-=1
                print("Did not find ", heuristic, " in trained model with ", len(heuristicPhrases), " heuristics")
            elif wordResults[3] == True and phraseResults[3] == True:
                totalCases-=1
                #print('No heuristic score for this message.')
            if score == correctScore:
                numRight+=1
            self.outputDf = self.outputDf.append({'Heuristic': heuristic, 'Message':message, 'Actual Score': correctScore, 'Predicted Overall Score': score}, ignore_index=True)
            self.phraseDf = self.phraseDf.append({'Heuristic': heuristic, 'Message':message, 'Actual Score': correctScore, 'Predicted Phrase Score': phraseResults[1]}, ignore_index=True)
            self.wordDf = self.wordDf.append({'Heuristic': heuristic, 'Message':message, 'Actual Score': correctScore, 'Predicted Word Score': wordResults[1]}, ignore_index=True)
        
        print(totalCases)
        print(numRight)
        answerPercent = (numRight/totalCases) * 100
        self.outputDf.to_excel('output.xlsx')
        self.phraseDf.to_excel('phraseOutput.xlsx')
        self.wordDf.to_excel('wordOutput.xlsx')
        testing = [answerPercent, x, y]
        print(testing)
        return testing

    def getOverallScore(self, wordResults, phraseResults, x, y):
        wordScore = wordResults[1]
        phraseScore = phraseResults[1]
        score = ((wordScore*x)+(phraseScore*y)) / 100
        return round(score)

