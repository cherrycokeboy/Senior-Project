import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('brown')
#nltk.download('universal_tagset')
text = nltk.word_tokenize("Intrepid Capital Management Has Lifted Its Position in Apple Com $AAPL by $493765 as Stock Value Declined; As Energy Recovery $ Share Value Declined Quantum Capital Management Has Lifted Stake by $655200".lower())
text = nltk.pos_tag(text)
news = nltk.corpus.brown.tagged_words(categories='news',tagset='universal')
tagFd = nltk.FreqDist(tag for (word,tag) in text)
print(tagFd.most_common())


