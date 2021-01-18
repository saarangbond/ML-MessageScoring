# ML-MessageScoring
## A machine learning model to score marketing messages based on their effectiveness.
- - -
This project is a machine learning model that is trained with marketing data and scores new marketing messages on a numeric scale based on their potential effectiveness on consumers.
This project utilizes the NLTK (Natural Language Tool Kit) library in python to process the English Language and give an understanding of marketing messages to a computer model.
- - -
### Phases:
  1. Build a structure to process data and store an effective training system within the computer model.
     - Sort each unique identifier into its own structure
     - Sort the words and phrases into those structures
     - Remove unwanted words and punctuation from the structures
  2. Pass the training structure to two separate models that assign separate scores based on a) word frequency and b) phrase frequency
  3. Garner the overall score based on the two separate model's scores.
- - -

### Running the Program
  This project was built and tested with Anaconda version 3.7 of Python.
  To run this program from the command line, you MUST use a version of Python 3 (Anaconda) with Pandas installed.

  TO RUN: use "python general.py" to run the default program with the sample data.

#### Command Line Arguments
  Multiple Command Line Argumants are optional for this program.

  * --data : takes in either "processed" or "raw" for multiple training data file dependencies.
  * --test : takes in either True or False to decide whether to run testing or not.
  * --trainingFile : specifies the training data file in this directory.
  * --testingFile : specifies the testing data file in this directory.
  * --wordFile : specifies an output file for the word storage structure.
  * --phraseFile : specifies an output file for the phrase storage structure.
- - -
### Libraries Used In This Project
  - NLTK :
    - nltk.tokenize : used for tokenizing bodies of text into individual words and punctuation.
    - nltk.stem : used for retrieving the stem for any word or retrieving the lemmatized form of a word.
    - nltk.corpus : used for fetching dictionaries and stopword lists to filter junk.
    - nltk.collocations : used for finding phrase combinations from a tokenized body of text.
  - Pandas : used for loading and sorting data with the use of dataframes.
  - argparse : used to parse command line arguments.
  - collections :
    - Counter : used for sorting through unordered data structures such as dictionaries.
  - progressbar : used to display a progress bar for reference during lengthy processes.
  - csv : used to write data from structures to csv files.
  - os : used to get current working directory.
  
