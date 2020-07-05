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
training.cleanLowSampleSizes(25)
training.cleanLowVariety(5)
data = training.getData()
heuristics = training.getHeuristics()

numRows = data.shape[0]
print('NumRows = ', numRows, '\n', data.dtypes)

print(data.head(15).transpose())

for k in heuristics.keys():
    most = max(heuristics[k].items(), key = lambda x : x[1])
    print(k,most)

testing = PrepTestingData()
testing.dropOrig()
testing_data = testing.getTestingData()

print(testing_data.head(5).transpose())

modelTesting = AccuracyTesting()

highest = 0
bestResults = []
for i in range(1,101):
    print('...')
    for j in range(1,101):
        for k in range(0,5):
            results = modelTesting.runAccuracyTest(heuristics, testing_data, i, j, k)
            print(results)
            if(results[0] > highest):
                highest = results[0]
                bestResults = results

print('Best Results:')
print(bestResults)