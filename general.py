import getopt
import sys
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from WordModel.wordModel import WordModel
from PhraseModel.phraseModel import PhraseModel
from overall import OverallScore
import unicodedata

print(unicodedata.unidata_version)

ps = PorterStemmer()
wordModel = WordModel()
phraseModel = PhraseModel()


args = sys.argv
test = True
loadRaw = None
if args[1] == "--ready":
    loadRaw = False
elif args[1] == "--raw":
    loadRaw == True
elif args[1] == "--rawnotest":
    loadRaw = True
    test = False
elif args[1] == "--readynotest":
    loadRaw = False
    test = False
else:
    loadRaw = None

testFile = 'TestingData.xlsx'

print('Automatically Score Heuristicized Messages: ', args[1])

if loadRaw == True or loadRaw == False:
    
    wordModel.loadTrainingData(loadRaw)
    wordModel.loadTestingData(testFile)
    heuristics = wordModel.getHeuristics()
    print('WORD MODEL PREP COMPLETE\n')

    phraseModel.loadTrainingData(loadRaw)
    phraseModel.loadTestingData(testFile)
    heuristicPhrases = phraseModel.getHeuristicPhrases()
    print('PHRASE MODEL PREP COMPLETE\n')
    
    if test == True:
        testing_data = wordModel.getTestingData()
        overallScore = OverallScore(testing_data)
        bestTesting = [0, 0, 0]
        for x in range(50,51):
            y = 100-x
            testing = overallScore.runScores(heuristicPhrases, heuristics, x, y)
            if(testing[0] > bestTesting[0]):
                bestTesting = testing
        print('Best: ', bestTesting)



