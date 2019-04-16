import tweepy
import json
import pandas as pd
import datetime
import requests
import time
import re
import pprint

def request_stuff(text):
    """makes a request call into a data frame, the request must be a csv"""
    r = requests.get(text)
    with open('temp.csv','w') as f:
        f.write(r.text)
    with open('temp.csv','r') as f:
        df = pd.read_csv(f)   
    return df


def get_stock_symbol(company):
    """Get a companies stock symbol. Based on a 80% match search. Returns None if can't find a symbol"""
    df = request_stuff('https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords='+company+'&apikey=37TEV7FAN4FPQ5RS&datatype=csv')
    done = False
    currentTest = 0
    stock_symbol = ''
    while not done:
        if len(df) <= currentTest:
            done = True
        elif df.iloc[currentTest,3] == 'United States' and df.iloc[currentTest,8] >= 0.6:
            stock_symbol = df.iloc[currentTest,0]
            done = True        
        else:
            currentTest += 1
    if stock_symbol == '':
        return None
    else:
        return stock_symbol
    
def stock_info(stock,date):
    """returns a dataframe with the date, open, high, low, close, and volume of the stock for each day"""
    df = request_stuff('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+stock+'&apikey=37TEV7FAN4FPQ5RS&datatype=csv')
    return df

def clean_tweet(tweet): 
        ''' Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. '''
        #return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", tweet).split())

def to_df(tweet):
    '''takes a dictionary of tweets and returns just the data and text of each tweet in a panda dataframe'''
    df = pd.DataFrame.from_dict(tweet,orient = 'index')
    return df.transpose()[['created_at','full_text']]
    
def get_tweets(stock,max_tweets,api):  
    '''is given a stock symbol string and a int of how many tweets are needed, and returns a dataframe with comlumns created at, text, and symbol for all returned tweets'''
    searched_tweets = [status for status in tweepy.Cursor(api.search, q=stock,languages=["en"],tweet_mode='extended').items(max_tweets)]
    #pp = pprint.PrettyPrinter(indent=4)
    #for i in searched_tweets:
        #pp.pprint(i._json)
        #print(i._json)
    dfMaster = pd.DataFrame()
    for i in searched_tweets:
        if i._json['lang'] == 'en':
            dfMaster = dfMaster.append(to_df(i._json))
    try:
        dfMaster['text'] = dfMaster['full_text'].apply(lambda x : clean_tweet(x))
        #print(dfMaster['text'])
    except:
        pass
    dfMaster['Company'] = stock
    return dfMaster

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
    companies["Search"] = companies["Symbol"].map(str) + ' ' + companies["Name"]
    return companies['Search']

def name_to_stock():
    names = get_company()
    for name in names:
        get_stock_symbol(name)
        

def change_to_u8(file):
    BLOCKSIZE = 1048576 # or some other, desired size in bytes
    with codecs.open(file, "r", "cp1254") as sourceFile:
        with codecs.open(file, "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents) 
                
                
def first_time(names):
    done = pd.Series(data = names)
    with open('done.csv','w') as f:
        f.write(done.to_csv())
    return done


def main():
    names = get_company().tolist()
    try:    
        with open('done.csv','r') as f:
            done = pd.Series.from_csv(f)
    except:
        done = first_time(names)
    oldDf = pd.read_csv('tweetReal.csv', names = ['id','created_at','full_text','text','Search'],lineterminator='\n')
    done = done.tolist()
    not_good = []
    count = 0
    for name in done:
        try:
            if name in done:
                oldTweets = oldDf[oldDf['Search'] == name]
                with open('tweetReal.csv','ab') as f:
                    df = get_tweets(name,500,api)
                    if "text" in df.columns:
                        df.drop_duplicates(subset ="text", keep = 'first', inplace = True)
                        df = pd.concat([df,oldTweets],sort=False)
                        df.drop_duplicates(subset = 'text',keep = 'first' ,inplace = True)
                        df = pd.concat([df,oldTweets],sort=False)
                        df.drop_duplicates(subset = 'text',keep = False ,inplace = True)
                    else:
                        not_good.append(name)
                        print(not_good)
                    f.write(df.to_csv(header=False).encode('utf-8'))
                with open('done.csv','w') as f:
                    f.write(pd.DataFrame(data=done).to_csv())
                count += 1
                print('Done With ' + name + " this is search number " + str(count))
        except Exception as inst:
            count = 0
            print(inst)
            print('sleeping')
            for i in range(15):
                print('Active in',i+1,'mintues')
                time.sleep(60)
            
auth = tweepy.OAuthHandler('wklaEQfhDVGXfy8rpfw1ko8tX' , 'w2eDqOgQGSWfWJASHM7SBYXTWspDi9NiIvkTlUrz4FW8i1aGMV' )
auth.set_access_token('1013820252-0uNb7vD9oY5YObc3KsyrhNdoRmyzRKQytXpZk0i', 'NnsjS4vXMccCuyEPgTdURj63MeVXLtCtJLncNjX3cy0LQ' )
api = tweepy.API(auth) 

#get_company()
#with open('tweets.csv','wb') as f:
#df = get_tweets("APPL",3,api)
#print(df['text'])
    #print(df.to_csv().encode('utf-8'))
    #f.write(df.to_csv(header=True).encode('utf-8'))
main()


#df = get_tweets('@apple',100)
#with open('tweets.csv','ab') as file: 
    #file.write(df.to_csv(header=False).encode('UTF-8'))