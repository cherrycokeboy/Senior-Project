import tweepy
import json
import pandas as pd
import datetime
import requests
import time
import re
import pprint

class tweetsDf:
    def __init__ (self, done,api,oldDf = 'tweetReal.csv', target = 'tweetReal.csv', col = ['id','created_at','full_text','text','Search']):
        self.target = target
        self.col = col
        self.companies = self.get_company().tolist()
        if type(done) == type(None):
            self.done = self.first_time(self.companies)
        else:
            self.done = done
        self.api = api
        self.oldDf = pd.read_csv(oldDf, names = self.col,lineterminator='\n')
    
    def first_time(self,names):
        done = pd.Series(data = names)
        with open('done.csv','w') as f:
            f.write(done.to_csv())
        return done    
    def get_company(self):
        companies = pd.read_csv('companylist.csv')
        companies['Name'] = companies['Name'].apply(self.get_rid)
        companies["Search"] = companies["Symbol"].map(str) + ' ' + companies["Name"]
        return companies['Search']    
    
    def clean_tweet(self,tweet): 
            ''' Utility function to clean tweet text by removing links, special characters 
            using simple regex statements. '''
            #return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 
            return ' '.join(re.sub("(@[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", tweet).split())    
    def to_df(self,tweet):
        '''takes a dictionary of tweets and returns just the data and text of each tweet in a panda dataframe'''
        df = pd.DataFrame.from_dict(tweet,orient = 'index')
        return df.transpose()[['created_at','full_text']]    
    
    def get_tweets(self,stock,max_tweets,api):  
        '''is given a stock symbol string and a int of how many tweets are needed, and returns a dataframe with comlumns created at, text, and symbol for all returned tweets'''
        done = False
        while not done:
            try:
                searched_tweets = [status for status in tweepy.Cursor(self.api.search, q=stock,languages=["en"],tweet_mode='extended').items(max_tweets)]
                done = True
            except Exception as inst:
                count = 0
                print(inst)
                print('sleeping')
                for i in range(15):
                    print('Active in',i+1,'mintues')
                    time.sleep(60)         
        dfMaster = pd.DataFrame()
        for i in searched_tweets:
            if i._json['lang'] == 'en':
                dfMaster = dfMaster.append(self.to_df(i._json))
        try:
            dfMaster['text'] = dfMaster['full_text'].apply(lambda x : self.clean_tweet(x))
        except:
            pass
        dfMaster['Company'] = stock
        return dfMaster
    def get_rid(self,x):
        x = x.lower()
        x = x.replace('inc','')
        x = x.replace('corp','')
        x = x.replace('.','')
        x = x.replace(',','')
        return x   
    def name_to_stock(self):
        names = get_company()
        for name in names:
            get_stock_symbol(name) 
            
    def dropDups(self,df,oldTweets):
        if "text" in df.columns:
            df.drop_duplicates(subset ="text", keep = 'first', inplace = True)
            df = pd.concat([df,oldTweets],sort=False)
            df.drop_duplicates(subset = 'text',keep = 'first' ,inplace = True)
            df = pd.concat([df,oldTweets],sort=False)
            df.drop_duplicates(subset = 'text',keep = False ,inplace = True)   
        return df
    
    def createTweets(self,max_tweets):
        count = 0
        for name in self.done:
            if type(name) != float:
                oldTweets = self.oldDf[self.oldDf['Search'] == name]
                
                df = self.get_tweets(name,max_tweets,self.api)
                df = self.dropDups(df,oldTweets)
                with open('done.csv','r') as f:
                    notDone = pd.read_csv(f).dropna()
                    notDone.columns = ['index','comp']
                notDone = notDone[notDone['comp'] != name].drop(['index'],axis=1)
                with open(self.target,'ab') as f:
                    f.write(df.to_csv(header=False).encode('utf-8'))
                with open('done.csv','w') as f:
                    f.write(notDone.to_csv())
                count += 1
                print('Done With ' + name + " this is search number " + str(count))






