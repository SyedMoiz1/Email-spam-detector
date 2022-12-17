import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

data = pd.read_csv("spam.csv", encoding="latin-1")#reading dataset

#Dataframe includes 3 unwanted columns Unnamed:2, Unnamed:3, Unnamed:4
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

#Since in the df, spam and ham are non numerical values, need to map spam and ham to 1 and 0 respectively
data['class']=data['class'].map({'ham':0, 'spam':1})#mapping func

#The messages must be represented as a numerical structure, use a count vectorizer to convert from text to numbers
cvectorizer = CountVectorizer()

X = data['message']
Y=data['class']

X = cvectorizer.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#use a naive bayes classifier
NBmodel = MultinomialNB()
NBmodel.fit(x_train, y_train)

print(NBmodel.score(x_test, y_test)) #accuracy is approx. 98%

#save current model into pickle format
pickle.dump(NBmodel, open("spam.pkl", "wb"))
pickle.dump(cvectorizer, open("vectorizer.pkl", "wb"))

#test model using string
message = input("Enter your mail here: ")

def is_spam(message):
    data = [message]
    vector = cvectorizer.transform(data).toarray()#saved into vector format
    result = NBmodel.predict(vector)
    if(result==1):
        result="The mail is spam"
    else:
        result = "The mail is not spam"
    print(result)

is_spam(message)