import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from accuracy import AccuracyTesting
from loadTraining import PrepTrainingData
from loadTesting import PrepTestingData
ps = PorterStemmer()

training = PrepTrainingData()
training.dropOrigAndLow()
training.sortHeuristics()
training.sortWords()
training.removeStopwords()
training.cleanLowSampleSizes(2)
training.cleanLowVariety(6)
data = training.getData()
#print(data.shape[0])
heuristics = training.getHeuristics()
heuristicCounts = training.getHeuristicCounts()
#print("Model has ", len(heuristics), "")

numRows = data.shape[0]
#print('NumRows = ', numRows, '\n', data.dtypes)

#print(data.head(15).transpose())

for k in heuristics.keys():
    most = max(heuristics[k].items(), key = lambda x : x[1])
    print(k,most)

testing = PrepTestingData()
testing.dropOrig()
testing_data = testing.getTestingData()

#print(testing_data.head(5).transpose())

modelTesting = AccuracyTesting()

#print(heuristicCounts)
#print("Model has ", len(heuristics), " heuristics")
highest = 0
bestResults = []
results = modelTesting.runAccuracyTest(heuristics, testing_data)


print('Results:', results)