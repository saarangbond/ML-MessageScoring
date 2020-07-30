import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()

class PrepTestingData:
    
    def __init__(self, testFile):
        testFile = '\\' + testFile
        self.testing_data = pd.read_excel(r'C:\Users\Saarang\Documents\NewristicsScoringModel' + testFile)
        print('TESTING DATA LOADED')

    def dropOrig(self):
        self.testing_data = self.testing_data.loc[self.testing_data['MessageWriter'] != 'Original']
        print('DROPPED ORIGINAL MESSAGES IN TESTING SET')
        self.testing_data.reset_index(drop=True, inplace = True)

    def getTestingData(self):
        return self.testing_data