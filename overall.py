import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from WordModel.wordAccuracy import WordAccuracyTesting
from PhraseModel.phraseAccuracy import PhraseAccuracyTesting
import statistics
ps = PorterStemmer()

class OverallScore:

    def __init__(self, testing_data):
        self.outputDf = pd.DataFrame(columns=["Sentiment","Message","Actual Score","Predicted Overall Score"])
        self.phraseDf = pd.DataFrame(columns=["Sentiment","Message","Actual Score","Predicted Phrase Score"])
        self.wordDf = pd.DataFrame(columns=["Sentiment","Message","Actual Score","Predicted Word Score"])
        self.phraseAccuracy = PhraseAccuracyTesting()
        self.wordAccuracy = WordAccuracyTesting()
        self.testing_data = testing_data

    def runScores(self, sentimentPhrases, sentiments, x, y, runWordAccuracy, runPhraseAccuracy):
        numRight = 0
        totalCases = self.testing_data.shape[0]
        for i in range(0,self.testing_data.shape[0]):
            sentiment = self.testing_data['SentimentName'][i]
            sentiment = sentiment.upper()
            message = self.testing_data['Messages'][i]
            message = message.lower()
            correctScore = self.testing_data['ManualScore'][i]
            wordResults = [None, None, None, None]
            phraseResults = [None, None, None, None]
            if runWordAccuracy == True:
                wordResults = self.wordAccuracy.runAccuracyTest(sentiments, sentiment, message, correctScore)
            if runPhraseAccuracy == True:
                phraseResults = self.phraseAccuracy.runAccuracyTest(sentimentPhrases, sentiment, message, correctScore)
            score = self.getOverallScore(wordResults, phraseResults, x, y)
            if wordResults[2] == True and phraseResults[2] == True:
                totalCases-=1
                print("Did not find ", sentiment, " in trained model with ", len(sentimentPhrases), " sentiments")
            elif wordResults[3] == True and phraseResults[3] == True:
                totalCases-=1
                #print('No sentiment score for this message.')
            if score == correctScore:
                numRight+=1
            self.outputDf = self.outputDf.append({'Sentiment': sentiment, 'Message':message, 'Actual Score': correctScore, 'Predicted Overall Score': score}, ignore_index=True)
            self.phraseDf = self.phraseDf.append({'Sentiment': sentiment, 'Message':message, 'Actual Score': correctScore, 'Predicted Phrase Score': phraseResults[1]}, ignore_index=True)
            self.wordDf = self.wordDf.append({'Sentiment': sentiment, 'Message':message, 'Actual Score': correctScore, 'Predicted Word Score': wordResults[1]}, ignore_index=True)
        
        print(totalCases)
        print(numRight)
        answerPercent = (numRight/totalCases) * 100
        self.outputDf.to_excel('score.xlsx')
        self.phraseDf.to_excel('phraseScore.xlsx')
        self.wordDf.to_excel('wordScore.xlsx')
        testing = [answerPercent, x, y]
        print(testing)
        return testing

    def getOverallScore(self, wordResults, phraseResults, x, y):
        wordScore = 0
        phraseScore = 0
        if wordResults[0] != None:
            wordScore = wordResults[1]
        if phraseResults[0] != None:
            phraseScore = phraseResults[1]
        score = ((wordScore*x)+(phraseScore*y)) / 100
        return round(score)

