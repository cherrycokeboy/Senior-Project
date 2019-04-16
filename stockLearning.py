from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import re
from nltk.corpus import stopwords
import nltk
import ast

def get_rid(x):
    x = x.lower()
    x = x.replace('inc','')
    x = x.replace('corp','')
    x = x.replace('.','')
    x = x.replace(',','')
    return x

def get_company():
    companies = pd.read_csv('companylist.csv')
    companies['Name'] = companies['Name'].apply(get_rid)
    companies["Symbol"] = companies["Symbol"].apply(get_rid)
    listy =  companies['Name'].to_list() + companies['Symbol'].to_list()
    returny = []
    for i in listy:
        returny += i.split()
    return returny
    


def createTrain_Test():
    with open('finalStock.csv','rb') as f:
        df = pd.read_csv(f)
        for col in df.columns:
            if "Unnamed" in col or "index" in col:
                df.drop(col,axis=1,inplace=True)
    train , test = train_test_split(df,test_size=.2, random_state = 420)
    with open('tweeterTrain.csv','wb') as f:
        f.write(train.to_csv().encode('utf-8'))
    with open('tweeterTest.csv','wb')as f:
        f.write(test.to_csv().encode('utf-8'))



def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words


def cleanData(text):
    noSpace = re.compile("[.,:\?,\"()\[\]$] | [0-9]")
    space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    text = [noSpace.sub("", line.lower()) for line in text]
    text = [space.sub(" ", line) for line in text]
    return text





def partsOfSpeech():
    trainTokens = pd.Series()
    testTokens = pd.Series()    
    for i in range((len(tweetsTrain)//40000) + 1): 
        trainTokens = pd.concat([trainTokens,tweetsTrain['text'][i*40000:(i+1)*40000].apply(lambda x : nltk.pos_tag(nltk.word_tokenize(x)))],ignore_index=True)
        print("done with rows: " + str(i*40000) +"-"+ str((i+1)*40000) + " for the train set")
    for i in range((len(tweetsTest)//40000) + 1):
        testTokens = pd.concat([testTokens, tweetsTrain['text'][i*40000:(i+1)*40000].apply(lambda x : nltk.pos_tag(nltk.word_tokenize(x)))],ignore_index=True)
        print("done with rows: " + str(i*40000) +"-"+ str((i+1)*40000) +" for the test set")

        with open('tweetTrainPOS.csv','wb') as f:
            f.write(trainTokens.to_csv().encode('utf-8'))
        with open('tweetTestPOS.csv','wb') as f:
            f.write(testTokens.to_csv().encode('utf-8'))
            


def getPOF(x,pof):
    #print(ast.literal_eval(x))
    if type(x) != float:
        fd = nltk.FreqDist(x)
        returnStr = ''
        for wt in ast.literal_eval(x):
            if pof in wt[1]:
                returnStr += wt[0] + ' '
        return returnStr
    return ''
    
    
  
  
  
def main():
    
    
    #createTrain_Test()
    #with open('tweeterTrain.csv','rb') as f:
        #tweetsTrain = pd.read_csv(f)[['text','Bull']]
    #with open('tweeterTest.csv','rb')as f:
        #tweetsTest = pd.read_csv(f)[['text','Bull']]
    #nltk.download('stopwords')
    #english_stop_words = stopwords.words('english')   
    #tweetsTrain.sort_values('Bull',inplace = True)
    #tweetsTest.sort_values('Bull',inplace = True)
    
    #tweetsTrain['text'] = pd.Series(cleanData(remove_stop_words(tweetsTrain['text'].dropna().to_list())))
    #tweetsTest['text'] = pd.Series(cleanData(remove_stop_words(tweetsTest['text'].dropna().to_list())))
    
    #tweetsTrain.dropna(inplace=True)
    #tweetsTest.dropna(inplace=True)    
    #partsOfSpeech()
    with open('trainPOS.csv','rb') as f:
        tweetsTrain = pd.read_csv(f)
    with open('testPOS.csv','rb') as f:
        tweetsTest = pd.read_csv(f)
        
    #with open('trainPOS.csv','wb') as f:
        #f.write(tweetsTrain.to_csv().encode('utf-8'))
    #with open('testPOS.csv','wb')as f:
        #f.write(tweetsTest.to_csv().encode('utf-8'))    
    tweetsTrain['verbs'] = tweetsTrain['pos'].apply(lambda x : getPOF(x,'V'))
    print(tweetsTrain['verbs'])
    cv = CountVectorizer(binary = True,stop_words = english_stop_words + get_company())
    cv.fit(tweetsTrain['verbs'])
    X = cv.transform(tweetsTrain['verbs'])
    XTest = cv.transform(tweetsTest['verbs'])
    
    #target = [0 if i < len(tweetsTrain[tweetsTrain['Bull'] == 0]) else 1 for i in range(len(tweetsTrain))]
    train, test, ytrain, ytest = train_test_split(X, tweetsTrain['Bull'].to_list(), train_size = 0.8)
    #for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=1)
    lr.fit(train, ytrain)
    print ("Accuracy for C=%s: %s" % (1, accuracy_score(ytest, lr.predict(test))))
    
    
    
    feature_to_coef = {
        word: coef for word, coef in zip(
            cv.get_feature_names(), lr.coef_[0]
        )
    }
    for best_positive in sorted(
        feature_to_coef.items(), 
        key=lambda x: x[1], 
        reverse=True)[:5]:
        print (best_positive)
        
    for best_negative in sorted(
        feature_to_coef.items(), 
        key=lambda x: x[1])[:5]:
        print (best_negative) 
        
main()