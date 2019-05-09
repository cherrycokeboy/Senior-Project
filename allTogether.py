import twittere
import getstock
import tweepy
import pandas as pd

def main():
    try:    
        with open('done.csv','r') as f:
            done = pd.Series.from_csv(f,header = 0).dropna()
    except:
        done = None     
    auth = tweepy.OAuthHandler('wklaEQfhDVGXfy8rpfw1ko8tX' , 'w2eDqOgQGSWfWJASHM7SBYXTWspDi9NiIvkTlUrz4FW8i1aGMV' )
    auth.set_access_token('1013820252-0uNb7vD9oY5YObc3KsyrhNdoRmyzRKQytXpZk0i', 'NnsjS4vXMccCuyEPgTdURj63MeVXLtCtJLncNjX3cy0LQ' )
    api = tweepy.API(auth) 
    print(api.rate_limit_status())
    df = twittere.tweetsDf(done, api, oldDf = 'moremoreTweets.csv',target='moremoreTweets.csv')
    df.createTweets(1500)
    #getstock.createStockDf('moreTweets.csv','stockTweets.csv')
main()