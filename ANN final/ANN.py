# Natural Language Processing
# Importing the libraries
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from textstat.textstat import textstat
nltk.download('stopwords')

dataset = pd.read_csv('outputFullSet3.csv', delimiter=",") #nrows
dataset = dataset.replace(np.nan, '-1', regex=True)
noOfRows = len(dataset.index)

mailDateCol = 3
subjectLineCol = 4
clientIdCol = 2
idCol = 1
deliveredNoCol = 5
openedNoCol = 6

sizeOfBagOfWords = 500

###############################################################################################

###############################################################################################
#Age of the subjectLine
# Season Sent
mailingDates = dataset.iloc[:, mailDateCol]
seasonSent = []
ageOfSline = []
now = datetime.datetime.now()
# DaySent & Day Opened
# Monday = 0, Sunday = 6
daySent = dataset.iloc[:, mailDateCol]
daySent = []
# Seasons
# Xmas&Newyear = 1, Halloween = 2, Vallentines Day = 3, Easter = 4

print("DATA SET COMPRISES OF: ", noOfRows, " ROWS")

for i in range(0, noOfRows):

    
    try:
        year = int(str(mailingDates[i])[2:6])
        month = int(str(mailingDates[i])[7:9])
        day = int(str(mailingDates[i])[10:12])
        date = datetime.date(year, month, day)
        daySent.append(date.weekday())
    
        age = datetime.date(now.year, now.month, now.day) - date
        ageOfSline.append(age.days)
    
        sLine = re.sub('[^a-zA-Z]', ' ', dataset['""subject""'][i])
        sLine = sLine.lower()
        sLine = sLine.split()
        ps = PorterStemmer()
        sLine = [ps.stem(word) for word in sLine if not word in set(stopwords.words('english'))]
        if (date.month == 12) and ("christma" in sLine or ("new" in sLine and "year" in sLine)):
            seasonSent.append(1)
        elif date.month == 10 and ("halloween" in sLine):
            seasonSent.append(2)
        elif date.month == 2 and ("vallentin" in sLine):
            seasonSent.append(3)
        elif (date.month == 4) and ("easter" in sLine):
            seasonSent.append(4)
        else:
            seasonSent.append(0)
    
    except:
        print("ERROR: ", dataset['""subject""'][i])
        
dataset['Season'] = seasonSent
dataset['Age(Days)'] = ageOfSline
dataset['daySent'] = daySent
###############################################################################################
print("FINISHED READING IN DATA")

###############################################################################################
# Counting numb of Specials
specialCharArray = []
sLines = dataset.iloc[:, subjectLineCol]
for sline in sLines:
    # print(sLine)
    count = 0
    for c in str(sline):
        if str(c) in (",", ".", "!", "?", "+", "&", "|", "-"):
            count = count + 1
    specialCharArray.append(count)

dataset['No of specialChar'] = specialCharArray
###############################################################################################

print("FINISHED SPECIAL CHARACTERS")


###############################################################################################

#Complexity
complexityArr = []
sLines = dataset.iloc[:, subjectLineCol]
complexity = []
for s in sLines:
    # if (len(s) < 10)
    try:
        complexityArr.append(textstat.flesch_reading_ease(s))
    except:
        print("ERROR THING: ", s)
        complexityArr.append(100.0)
dataset['Complexity'] = complexityArr

###############################################################################################

print("FINISHED COMPLEXITY")

###############################################################################################
# Counting numb of Propositions
noOfPropositonsArr = []
propositonArr = []
sLines = dataset.iloc[:, subjectLineCol]
for s in sLines:
    propTally = 0

    input = s
    input = input.lower()
    input = re.sub('[^A-Za-z0-9]+', ' ', input)

    propositionList = ["best", "better", "good", "most", "top", "biggest", "bigger",
                       "big", "cheapest", "cheaper", "cheap", "highest", "higher", "amazing"
                       "high", "highly", "lowest", "lower", "low", "limited", "exclusive",
                       "gold", "silver", "platinum","bronze", "deal", "unheard", "price", "offer",
                       "smallest", "sav", "save", "saving", "savings", "sale"]

    inputArray = input.split(" ")
    ps = PorterStemmer()
    inputArray = [word for word in inputArray if not word in set(stopwords.words('english'))]


    propositionNumber = 0

    for i in range(len(inputArray)):
        word = inputArray[i]
        if word.lower() in propositionList:
           if i > 0:
               if inputArray[i-1].lower() not in propositionList:
                   propositionNumber += 1
           else:
               propositionNumber += 1

    # print(propositionNumber)
    noOfPropositonsArr.append(propositionNumber)

dataset['No of Propositons'] = specialCharArray

###############################################################################################

print("FINISHED PREPOSITIONS")

###############################################################################################
# Ratio of Subscribers Sent to max
# No sent already in a column
# Global max

ids = (dataset.iloc[:, idCol]).values

sentAmount = (dataset.iloc[:, deliveredNoCol]).values
openedAmount = (dataset.iloc[:, openedNoCol]).values
sentAmountArr = [0] * len(sentAmount)
openedAmountArr = [0] * len(openedAmount)

# Cleaning Data
for i in range(0, len(sentAmount)):
    try:
        if sentAmount[i] != "-1":
            sentAmountArr[i] = (sentAmount[i])[2:-2]
            sentAmount[i] = (sentAmount[i])[2:-2]
            sentAmountArr[i] = int(sentAmountArr[i])
            sentAmount[i] = int(sentAmount[i])
        if openedAmount[i] != "-1":
            openedAmountArr[i] = (openedAmount[i])[2:-2]
            openedAmount[i] = (openedAmount[i])[2:-2]
            openedAmountArr[i] = int(openedAmountArr[i])
            openedAmount[i] = int(openedAmount[i])
    except:
        print("ERROR: ", ids[i], sentAmount[i] )
        openedAmount[i] = "-1"
        sentAmount[i] = "-1"


#dataset = dataset.drop(rowsToDelete)

#GlobalMax
globalMax = np.amax(sentAmountArr)

# Client Max
clientList = (dataset.iloc[:, clientIdCol]).values

# Cleaning Data
for i in range(0, len(clientList)):
    clientList[i] = (clientList[i])[2:-2]

#clientMaxArr = []


#for i in range(0, len(set(clientList))):
#    currentClient = dataset.loc[(dataset['""client_id""']) == (list(set(clientList)))[i]]
#    max = -1;
#    for j in currentClient['""delivered""']:
#        if int(j) > int(max):
#            max = j
#    clientMaxArr.append([list(set(clientList))[i], max])
###############################################################################################

print("FINISHED SENT")

###############################################################################################
# Subjectline Proccesing
# Cleaning the texts
corpus = []
for i in range(0, noOfRows):
    # if dataset['Review'][i] == "":
    #     dataset['Review'][i] = " "
    review = re.sub('[^a-zA-Z]', ' ', dataset['""subject""'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = (sizeOfBagOfWords))
X = cv.fit_transform(corpus).toarray()

opened = dataset.iloc[:, openedNoCol].values
delivered = dataset.iloc[:, deliveredNoCol].values

###############################################################################################

print("FINISHED BAG OF WORDS")

#dataset.to_csv('output.csv')

###############################################################################################
# CREATE DATA SET WITH ONLY WHAT WE NEED

d = {}
df = pd.DataFrame(data = d)

df["Season"] = dataset.iloc[: ,7]
df["Age in Days"] = dataset.iloc[: ,8]
df["Day Sent"] = dataset.iloc[: ,9]
df["Number of Special Characters"] = dataset.iloc[: ,10]
df["Complexity"] = dataset.iloc[: ,11]
df["Number of Prepositions"] = dataset.iloc[: ,12]

# now add bag of words

collumnsList = []
for i in range(0, sizeOfBagOfWords):
    collumnsList.append(str(i))
wordsdf = pd.DataFrame(X, columns = collumnsList)

for i in range(0, sizeOfBagOfWords):
    df[str(i)] = wordsdf.iloc[:,i]
openrate = []

#Cleaning up opened and delivered and calculating openrate
for i in range(0, len(opened)):
    opened[i] = (opened[i])
    delivered[i] = (delivered[i])
    if int(delivered[i]) != 0:
        openrate.append(float(int(opened[i]))/float(int(delivered[i])))
    else:
        openrate.append(0)

# NOW SPLIT THE DATA INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, openrate, test_size = 0.1, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# MAKE A DATA FRAME OF Y
dy = pd.DataFrame(data = {})
dy["Click Rate"] = y_test

###############################################################################################
#Outputs
#X = bagOfWordsModel
#Y = Corresponding id value.abs

#clientMaxArr = max for each clientList
#globalMax = globalMax
#Day sent column
#Season column
#Numb of specialChar column

## THE ACTUAL ANN

print("FINISHED ALL DATA PREPROCESSING")

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import losses

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=(sizeOfBagOfWords+6), units=50, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units = 100, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units = 1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['mae'])
classifier.fit(X_train, y_train, batch_size = 100, epochs = 100)
scores = classifier.evaluate(X_test, y_test, verbose=0)

print()
print("------------------------------------------")
print("ACCURACY:", round(scores[1]*100, 3), "%")

from keras.models import load_model
classifier.save('model.h5')


print()
print()

###############################################################################

inputLine = "Travel to Morocco!"
season = 0
ageInDays = 1
daySent = 2


numberOfSpecialCharacters = 0

for c in str(inputLine):
    if str(c) in (",", ".", "!", "?", "+", "&", "|", "-"):
        numberOfSpecialCharacters = numberOfSpecialCharacters + 1



input = inputLine
input = input.lower()
input = re.sub('[^A-Za-z0-9]+', ' ', input)

propositionList = ["best", "better", "good", "most", "top", "biggest", "bigger",
                   "big", "cheapest", "cheaper", "cheap", "highest", "higher",
                   "high", "highly", "lowest", "lower", "low", "limited", "exclusive",
                   "gold", "silver", "bronze", "deal", "unheard", "price", "offer",
                   "smallest", "sav", "save", "saving", "savings", "sale"]

inputArray = input.split(" ")
ps = PorterStemmer()
inputArray = [word for word in inputArray if not word in set(stopwords.words('english'))]

numberOfPrepositions = 0

for i in range(len(inputArray)):
    word = inputArray[i]
    if word.lower() in propositionList:
       if i > 0:
           if inputArray[i-1].lower() not in propositionList:
               numberOfPrepositions += 1
       else:
           numberOfPrepositions += 1


complexity = textstat.flesch_reading_ease(input)

outputThing = cv.transform([inputLine]).toarray()[0]

d = [season, ageInDays, daySent, numberOfSpecialCharacters, numberOfPrepositions, complexity]

for i in range(len(outputThing)):
    d.append(outputThing[i])



d = {}
preprocessedInput = pd.DataFrame(data = d)
preprocessedInput["Season"] = [season]
preprocessedInput["Age in Days"] = [ageInDays]
preprocessedInput["Day Sent"] = [daySent]
preprocessedInput["Number of Special Characters"] = [numberOfSpecialCharacters]
preprocessedInput["Complexity"] = [complexity]
preprocessedInput["Number of Prepositions"] = [numberOfPrepositions]

# now add bag of words

collumnsList = []
for i in range(0, sizeOfBagOfWords):
    collumnsList.append(str(i))
words = pd.DataFrame(X, columns = collumnsList)

for i in range(0, sizeOfBagOfWords):
    preprocessedInput[str(i)] = outputThing[i]


predictions = classifier.predict(preprocessedInput)

print("SUBJECT LINE:", inputLine)
print("ESTIMATED CLICK RATE:", round(predictions[0][0] * 100, 3), "%")
