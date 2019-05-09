import requests
import datetime
import pandas as pd
import time
from bs4 import BeautifulSoup as bs
def scrape(webpage):
    COLUMNS = ['date','open','high','low', 'close','Adj Close','Volume']
    table = webpage.find("table") # Find the "table" tag in the page
    rows = table.find_all("tr")  # Find all the "tr" tags in the table
    cy_data = [] 
    for row in rows:
        cells = row.find_all("td") #  Find all the "td" tags in each row 
        cells = cells[0:7] # Select the correct columns (1 & 2 as python is 0-indexed)
        cy_data.append([cell.text.strip() for cell in cells]) # For each "td" tag, get the text inside it
    return pd.DataFrame(cy_data, columns=COLUMNS).drop(0, axis=0).drop_duplicates('date')

def get_webpage(url):
    response = requests.get(url)  #  Get the url
    return scrape(bs(response.text, 'html.parser')) #

def request_stuff(text):
    """makes a request call into a data frame, the request must be a csv"""
    r = requests.get(text)
    with open('temp.csv','w') as f:
        f.write(r.text)
    with open('temp.csv','r') as f:
        df = pd.read_csv(f)   
    return df

def stock_info(stock):
    """returns a dataframe with the date, open, high, low, close, and volume of the stock for each day"""
    df = get_webpage('https://finance.yahoo.com/quote/'+stock+'/history?p='+ stock)
    return df
def add_info(tweets,neededStock,sameCount):
    tweets.at[sameCount,"open"] = neededStock['open'].replace(',','').replace('-','0')
    tweets.at[sameCount,"high"] = neededStock['high'].replace(',','').replace('-','0')
    tweets.at[sameCount,"low"] = neededStock['low'].replace(',','').replace('-','0')
    tweets.at[sameCount,"close"] = neededStock['close'].replace(',','').replace('-','0')
    #if neededStock['Volume'] == '-':
        #neededStock['Volume'] = '0'
    tweets.at[sameCount,"volume"] = neededStock['Volume'].replace(',','').replace('-','0')
    tweets.at[sameCount,'increase'] = float(tweets.at[sameCount,"open"]) - float(tweets.at[sameCount,"close"])
    if tweets.at[sameCount,'increase'] > 0:
        tweets.at[sameCount,'Bull'] = 1
    else:
        tweets.at[sameCount,'Bull'] = 0


def createStockDf(start, target):
    #custom = holidays.HolidayBase()
    #hol = [date(2019,1,1),date(2019,1,21),date(2019,2,18),date(2019,4,19),date(2019,5,27),date(2019,7,4),date(2019,9,2),date(2019,10,14),date(2019,11,28),date(2019,12,25)]
    #for i in hol:
        #custom.append(i)    
    with open(start,'rb') as f:
        tweets = pd.read_csv(f,error_bad_lines=False)
    tweets.drop(tweets.columns[0],axis=1,inplace = True)
    if len(tweets.columns) == 5:
        tweets.columns = ['time','text','increase','comp','Bull']
        tweets = tweets.sort_values(by = 'comp').reset_index(drop=True)
    else:
        tweets.columns = ['time','badtext','text','comp']
        tweets['open'] = 0.0
        tweets['high']= 0.0  
        tweets['low'] = 0.0
        tweets['close']= 0.0        
        tweets['volume'] = 0.0      
        tweets['increase'] = 0.0
        tweets['Bull']= 0.0
    tweets = tweets.sort_values(by = 'comp').reset_index(drop=True)
    done = False
    company = tweets['comp']
    count = 0
    #with open('testing.csv','wb') as f:
        #f.write(tweets.to_csv().encode('utf-8'))        
    while not done:
        currentComp = tweets.iloc[count]['comp']
        try:
            currentSplit = currentComp.split()
        except:
            print(currentComp)
        same = True
        if len(currentSplit) >= 2:
            try:
                symbol = currentSplit[0]
                stock = stock_info(symbol)
                stock = stock.set_index('date')
                stock.sort_values(by = 'date')
            except Exception as inst:
                same = False
                print(inst)
                print('sleeping')
                for i in range(6):
                    print('Active in',i+1,'mintues')
                    time.sleep(10)                
            while same:
                timeWeNeedMinusOne = tweets.iloc[count]['time']
                datetimeMinusOne = datetime.datetime.strptime(timeWeNeedMinusOne,"%a %b %d %H:%M:%S +0000 %Y")
                datetimeNeeded = datetimeMinusOne + datetime.timedelta(1)
                #print(datetimeMinusOne, datetimeNeeded)
                if datetimeNeeded.weekday() == 5:
                    datetimeNeeded += datetime.timedelta(2)
                if datetimeNeeded.weekday() == 6:
                    datetimeNeeded += datetime.timedelta(1)
                find = False
                doIt = True
                while not find:
                    try:
                        find = True
                        neededStock = stock.loc[datetimeNeeded.strftime("%b %d, %Y")]
                        doIt = True
                    except:
                        lastStock = datetime.datetime.strptime(stock.index[0],"%b %d, %Y")
                        today = datetime.datetime.today()
                        if datetime.date(lastStock.year,lastStock.month,lastStock.day) + datetime.timedelta(1) <  datetime.date(today.year,today.month,today.day):
                            print(datetime.date(lastStock.year,lastStock.month,lastStock.day),  datetime.date(today.year,today.month,today.day))
                            find = True
                            doIt = False
                            same = False
                        elif datetimeNeeded > datetime.datetime.today():
                            find = True
                            doIt = False
                        else:
                            find = False
                            datetimeNeeded += datetime.timedelta(1)
                        #print(datetimeNeeded)
                        #count += 1
                        #print('didn\'t find')
                if doIt:
                    add_info(tweets,neededStock,count)
                    count += 1
                    if count > len(tweets):
                        done = True
                        tweets.at[count-1,"comp"] = currentSplit[0]
                        same = False
                        #with open(target,'wb') as f:
                            #f.write(tweets.to_csv().encode('utf-8'))
                        with open(target,'ab') as f:
                            f.write(tweets.to_csv().encode('utf-8'))                            
                    elif currentComp == tweets.iloc[count]['comp']:
                        tweets.at[count-1,"comp"] = currentSplit[0]
                    else:
                        tweets.at[count-1,"comp"] = currentSplit[0]
                        same = False
                        with open(target,'wb') as f:
                            f.write(tweets[['time','text','increase','comp','Bull']].to_csv().encode('utf-8'))
                            print('done with '+currentComp)
                else:
                    count += 1
                
        else:
            count += 1

#createStockDf('stockTweets.csv','stockTweets.csv')    