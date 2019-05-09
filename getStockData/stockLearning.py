from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag, map_tag, pos_tag_sents
from nltk.stem import WordNetLemmatizer
import ast
#nltk.download('wordnet')
import unicodedata
from unidecode import unidecode

count = 0
def deEmojify(inputString):
    returnString = ""

    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            replaced = unidecode(str(character))
            if replaced != '':
                returnString += replaced
            else:
                try:
                    returnString += "[" + unicodedata.name(character) + "]"
                except ValueError:
                    returnString += "[x]"

    return returnString
def posLogic (verbcount,adjcount,advcount,apcount):
    if verbcount == 0:
        verbcount = 1
        verb = True
    else:
        verbcount = 0
        verb = False
    if adjcount == 0 or adjcount == 1:
        adjcount += 1
        adj = True
    elif adjcount == 3:
        adjcount = 0
        adj = False
    else:
        adjcount += 1
        adj = False
    if advcount in [0,1,2,3]:
        advcount += 1
        adv = True
    elif advcount == 7:
        advcount = 0
        adv = False
    else:
        advcount += 1
        adv = False 
    if apcount in [0,1,2,3,4,5,6,7]:
        apcount += 1
        ap = True
    elif apcount == 15:
        apcount = 0
        ap = False
    else:
        apcount += 1
        ap = False
    return verbcount, verb, adjcount, adj, advcount, adv, apcount, ap 


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
    companies["Search"] = (companies["Symbol"].map(str) + ' ' + companies["Name"])
    search = companies['Search'].to_list()
    returnString = ''
    for word in search:
        returnString += ' ' + word 
    return returnString.split()
    
    #for i in range(len(search)):
        #returny[search[i][0]] = search[i][1]
        #for j in range(2,len(search[i])):
            #returny[search[i][0]] += ' ' + search[i][j]

    


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
    #nltk.download('stopwords')
    english_stop_words = stopwords.words('english')     
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words]))
    return removed_stop_words


def cleanData(text):
    noSpace = re.compile("_")
    brackets = re.compile("([\[][A-Za-z0-9 ]*[\]]+)")
    space = re.compile("([!,:;?,\"()\[\]+#$@&*%=|])|(<br\s*/><br\s*/>)|(\-)|(\/)")
    periodRemove = re.compile("[.]+")
    atSign = re.compile('[&][A-Za-z0-9]+')
    dollarSign = re.compile('[$][A-Za-z0-9]+')
    incWord = re.compile(' inc +| corp +| sec +| eps +| cbt +| lt +| gt +')
    andAmp = re.compile('H&amp;S|&amp')
    allCapital = re.compile('( [A-Z][A-Z]* )+')
    numbers = re.compile('([0-9])+')
    #quoteS = re.compile('([\'][s])')
    singleLetter = re.compile(' [a-z] +')
    newText = []
    lemma = WordNetLemmatizer()
    for line in text:
        line = deEmojify(line)
        line = brackets.sub(" ",line)
        line = allCapital.sub(" ",line)
        line = andAmp.sub("",line)
        line = atSign.sub("", line.lower())
        line = dollarSign.sub("", line.lower())
        line = noSpace.sub("", line.lower())
        line = space.sub(" ", line)
        line = incWord.sub(" ",line)
        line = numbers.sub("",line)
        #line = quoteS.sub(" is ",line)
        line = singleLetter.sub(" ",line)
        line = line.split(".")
        newLine = []
        for word in line:
            newLine.append(periodRemove.sub(" ",word))
        #line = " " .join(newLine)
        newText.append(line)
                           
        #text = [atSign.sub(" ", line.lower()) for line in text]
        #text = [noSpace.sub("", line.lower()) for line in text]
        #text = [space.sub(" ", line) for line in text]
        #text = [line.decode("utf-8-sig") for line in text]
        #text = [line.replace(u"\ufffd"," ") for line in text]
    return newText





def partsOfSpeech(tweetsTrain,tweetsTest):
    trainTokens = pd.Series()
    testTokens = pd.Series()    
    for i in range((len(tweetsTrain)//40000) + 1): 
        trainTokens = pd.concat([trainTokens,tweetsTrain['text'][i*40000:(i+1)*40000].apply(lambda x : nltk.pos_tag_sents(nltk.word_tokenize(x),tagset='universal'))],ignore_index=True)
        #.apply(lambda x : [(word, map_tag('en-ptb', 'universal',tag)) for word, tag in x])]
        print("done with rows: " + str(i*40000) +"-"+ str((i+1)*40000) + " for the train set")
    with open('tweetTrainPOS.csv','wb') as f:
        f.write(trainTokens.to_csv().encode('utf-8'))
    if tweetsTest != None:
        for i in range((len(tweetsTest)//40000) + 1):
            testTokens = pd.concat([testTokens, tweetsTrain['text'][i*40000:(i+1)*40000].apply(lambda x : nltk.pos_tag_sents(nltk.word_tokenize(x),tagset='universal'))],ignore_index=True)
            print("done with rows: " + str(i*40000) +"-"+ str((i+1)*40000) +" for the test set")
        with open('tweetTestPOS.csv','wb') as f:
            f.write(testTokens.to_csv().encode('utf-8'))
            


def getPOF(x,verb,adj,adv,adp):
    #print(ast.literal_eval(x))
    comp = get_company()
    #global count
    #print(count)
    #count += 1
    if type(x) != float:
        fd = nltk.FreqDist(x)
        returnStr = ''
        for wt in ast.literal_eval(x):
            #print(wt[0],wt[1])
            #if wt[1] not in get_company():
            if verb and 'VERB' in wt[1] :
                returnStr += wt[0] + ' '
            if 'ADJ' in wt[1] and adj:
                returnStr += wt[0] + ' '
            if 'ADV' in wt[1] and adv:
                returnStr += wt[0] + ' '
            if 'ADP' in wt[1] and adp:
                returnStr += wt[0] + ' '
        return returnStr
    return ''

def findStuff(row,comp):
    try:
        correctComp = comp[str(row['comp'].lower())]
        returny = any(substring in row['text'].lower() for substring in correctComp)
    except:
        returny = False
    return returny
    


def basicTopicIdentifier(df):
    df = df[0:50000]
    companies = get_company()
    #for segments in companies:
        #listOfCompanies += segments.split()
    df.dropna(inplace = True)
    #print(companies)
    df['Value'] = df.apply(lambda row: findStuff(row,companies) , axis=1)
    with open('justCheckingThings.csv','wb') as f:
        f.write(df.to_csv().encode('utf-8'))    
    print(df.Value.value_counts())
    
  
  
def main():
    
    
    ##createTrain_Test()
    with open('tweeterTrain.csv','rb') as f:
        master = pd.read_csv(f)[['text','Bull','comp']]
    with open('stockTweets.csv','rb') as f:
        master2point0 = pd.read_csv(f)[['text','Bull','comp']]
    with open('tweeterTest.csv','rb')as f:
        test = pd.read_csv(f)[['text','Bull','comp']]
  
    #master.sort_values('Bull',inplace = True)
    master = pd.concat([master,master2point0,test])
    master.sort_values('Bull',inplace = True)
    
    ##master = basicTopicIdentifier(master)
    master['text'] = pd.Series(cleanData((master['text'].dropna().to_list())))
    ##test['text'] = pd.Series(cleanData((test['text'].dropna().to_list())))
    aster.dropna(inplace=True)
    ##test.dropna(inplace=True)    
    train, test, ytrain, ytest = train_test_split(master['text'], master['Bull'], train_size = 0.8)
    train = pd.concat([train,ytrain],axis = 1)
    test = pd.concat([test,ytest],axis = 1)
    
    #with open('allDataWithOutPOSTrain.csv','rb') as f:
        #master = pd.read_csv(f)[['text','Bull']]    
    with open('allDataWithOutPOSTrain.csv','wb') as f:
        f.write(train.to_csv().encode('utf-8'))
    with open('allDataWithOutPOSTest.csv','wb')as f:
        f.write(test.to_csv().encode('utf-8')) 
    partsOfSpeech(master[['text']],test[['text']])

    
    with open('tweetTrainPOS.csv','rb') as f:
        trainpos = pd.read_csv(f,index_col = 0 , names = ['index','pos'])
    with open('tweetTestPOS.csv','rb') as f:
        testpos = pd.read_csv(f,index_col = 0 , names = ['index','pos'])
        
    #with open('trainPOS.csv','wb') as f:
        #f.write(tweetsTrain.to_csv().encode('utf-8'))
    #with open('testPOS.csv','wb')as f:
        #f.write(tweetsTest.to_csv().encode('utf-8')) 
    #verbcount = 0
    #adjcount =0
    #advcount = 0
    #apcount = 0
    #for i in range (16):
    #verbcount, verb, adjcount, adj, advcount, adv, apcount, ap = posLogic(verbcount,adjcount,advcount,apcount)
    partOfTrainpos = pd.concat([trainpos['pos'], master['Bull']], axis=1).reset_index()
    with open('universalTrain.csv','wb') as f:
        f.write(partOfTrainpos.to_csv(index = 0).encode('utf-8'))
    #with open('universalTrain.csv','rb') as f:
        #partOfTrainpos = pd.read_csv(f,index_col = 0 )
    #print(partOfTrainpos)
    partOfTrainpos['words'] = partOfTrainpos['pos'].apply(lambda x : getPOF(x,True,True,True,True)).apply(lambda x: str.strip(x))
    master = partOfTrainpos[['words','Bull']]
    master = master[master['words'] != '']
    master = master.dropna()
    
    #with open('temp1.csv','wb') as f:
        #f.write(master.to_csv(index = 0).encode('utf-8')) 
    #with open('temp1.csv','rb') as f:
        #master = pd.read_csv(f, names = ['words','Bull'])
    #print(master.columns)
        
    english_stop_words = stopwords.words('english')
    
    #cv = CountVectorizer(binary = False)
    tv = TfidfVectorizer(stop_words = english_stop_words)
    tv.fit(master['words'])
    X = tv.transform(master['words'])
    #with open('temp.csv','wb')as f:
        #f.write(master.to_csv().encode('utf-8'))      
    #target = [0 if i < len(tweetsTrain[tweetsTrain['Bull'] == 0]) else 1 for i in range(len(tweetsTrain))]
    train, test, ytrain, ytest = train_test_split(X, master['Bull'].to_list(), train_size = 0.8)
    for i in [0.001,0.005,0.025,0.05,0.1]:
        print("Running test for " + str(i) +"")
        sv = LinearSVC(C=i)
        #lr = LogisticRegression(C=i)
        sv.fit(train, ytrain)
        print ("Accuracy for C=%s: %s" % (i, accuracy_score(ytest, sv.predict(test))))



    feature_to_coef = {
        word: coef for word, coef in zip(
            tv.get_feature_names(), sv.coef_[0]
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