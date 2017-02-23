import nltk
import numpy
import re
import time
from nltk.util import ngrams
import csv
import pickle

def preprocess():
	## Add print statements as status updates
    print('preprocessing')
    global source_text
    source_text = [text.lower() for text in source_text]
    
    print('tokenizing')
    tokenized_text = [nltk.word_tokenize(text) for text in source_text]
    
    print('stemming')
    ## Stem needs to be in for loop to allow exception handling
    porter = nltk.PorterStemmer()
    global stemmed_text
    stemmed_text = []
    for tokens in tokenized_text:
        stemmed_line = []
        for t in tokens:
            try:
                stemmed_line.extend([porter.stem(t)])
            except IndexError:
                stemmed_line.extend('')
        stemmed_text.append(stemmed_line)
    
    print('removing rare words')
    vocab = nltk.FreqDist(w for line in stemmed_text for w in line)
    rarewords_list = set(vocab.hapaxes())
    stemmed_text = [['<RARE>' if w in rarewords_list else w for w in line] for line in stemmed_text]

    ## Pickle tokenized text for lexical ngrams
    ## stemmed text for pos ngrams
    ## source text for regex function
    pickle.dump(stemmed_text, open('stemmed_text.pkl','wb'))
    pickle.dump(source_text, open('source_text.pkl', 'wb'))
    pickle.dump(tokenized_text, open('tokenized_text.pkl','wb'))

def make_ngrams(text, ngramCount, pos = False):
	## if looking for POS, tag words, return word POS
    if pos:
        tag_text = nltk.pos_tag(text)
        text = [tag for (word, tag) in tag_text]

    ## nltk ngram() returns generator, convert to list
    text_ngs = list(ngrams(text, ngramCount))

    return text_ngs

def most_common(n_grams, n):
	## Generate FreqDist of ngrams
    fd = nltk.FreqDist(n_grams)
    ## Return the specified number of most common ngrams
    common = fd.most_common(n)
    
    ## FreqDist returns tuples of ngrams with counts, return only the ngrams
    return [a for (a,b) in common]

def most_common_ngrams(text, ngramCount, mostCommonCount, pos = False):
	## This is inefficient--shouldn't have 'if pos' in two different functions
    if pos:
        n_grams = [make_ngrams(doc, ngramCount, pos = pos) for doc in text]
    
    ## If unigrams, return ngrams if pos
    if ngramCount == 1:
    	if pos:
    		return n_grams

    	## If lexical unigrams, return unigrams without stopwords
    	text = [[word for word in essay if word not in nltk.corpus.stopwords.words('english')] for essay in text]
    	## return bag of words
    	return text

    ## if n > 1, create ngrams
    n_grams = [make_ngrams(doc, ngramCount) for doc in text]
        
    ## Flatten list of lists into one list for FreqDist 
    all_ngrams = [ng for ng_list in n_grams for ng in ng_list]

    ## Find n most common ngrams
    common_ngrams = most_common(all_ngrams, mostCommonCount)
 
    return common_ngrams

def bag_of_function_words():
	bow = []
	for sw in nltk.corpus.stopwords.words('english'):
		counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, line)) for line in source_text]
		bow.append(counts)
	return bow, nltk.corpus.stopwords.words('english')

## pos ngrams--use tokenized text
def bag_of_pos_trigrams():
	## Initialize bag of words, add print statements and time updates
	print('pos trigrams')
	trigram_pos_bow = []
	print('top 500 pos trigrams start: ', time.time())
	## find top 500 trigrams
	top_500_trigrams = most_common_ngrams(tokenized_text, 3, 500, pos = True)
	print('top 500 pos trigrams done: ', time.time())
	print ('bag of pos trigrams start: ' , time.time())

	## Looping through each of the top trigrams, return list of len 500, each sublist is len 570
	for trigram in top_500_trigrams:
		counts = [sum(trigram == n for n in make_ngrams(essay, 3, pos = True))/len(essay) for essay in tokenized_text]
		trigram_pos_bow.append(counts)

	print ('bag of pos trigrams done: ' , time.time())

	return trigram_pos_bow, top_500_trigrams

## Other ngram features follow same pattern
def bag_of_pos_bigrams():
	print('pos bigrams')
	bigram_pos_bow = []
	print('top 100 pos bigrams start: ', time.time())
	top_100_bigrams = most_common_ngrams(tokenized_text, 2, 100, pos = True)
	print('top 100 pos bigrams done: ', time.time())

	print('pos bigrams start: ', time.time())
	for bigram in top_100_bigrams:
		counts = [sum(bigram == n for n in make_ngrams(essay, 2, pos = True))/len(essay) for essay in tokenized_text]
		bigram_pos_bow.append(counts)
	print('pos bigrams done: ', time.time())

	return bigram_pos_bow, top_100_bigrams

def bag_of_pos_unigrams():
	print('pos unigrams')
	unigram_pos_bow = []
	print('get unigrams pos start: ', time.time())
	## n most common number is empty argument for returning all unigrams
	all_unigrams = most_common_ngrams(tokenized_text, 1, 500, pos = True)
	print('get unigrams pos done: ', time.time())

	print('unigrams pos start: ', time.time())
	for unigram in all_unigrams:
		counts = [sum(unigram == n for n in make_ngrams(essay, 1, pos = True))/len(essay) for essay in tokenized_text]
		unigram_pos_bow.append(counts)
	print('unigrams pos done: ', time.time())

	return unigram_pos_bow, all_unigrams

## Lexical ngrams--use stemmed text
def bag_of_trigrams():
	print('trigrams')
	trigram_bow = []
	print('top 500 trigrams start: ', time.time())
	top_500_trigrams = most_common_ngrams(stemmed_text, 3, 500)
	print('top 500 trigrams done: ', time.time())

	print ('bag of trigrams start: ' , time.time())
	for trigram in top_500_trigrams:
		counts = [sum(trigram == n for n in make_ngrams(essay, 3))/len(essay) for essay in stemmed_text]
		trigram_bow.append(counts)
	print ('bag of trigrams done: ', time.time())

	return trigram_bow, top_500_trigrams

def bag_of_bigrams():
    bigram_bow = []
    top_100_bigrams = most_common_ngrams(stemmed_text, 2, 100)
    print('counting common bigram start: ', time.time())
    #print("TOP BIGRAM: ", top_100_bigrams[0])
    for bigram in top_100_bigrams:
  #      print('BIGRAM: ',bigram)
        counts = [sum(bigram == n for n in make_ngrams(essay, 2))/len(essay) for essay in stemmed_text]
        bigram_bow.append(counts)
    print('counting common bigram done: ', time.time())

    return bigram_bow, top_100_bigrams

def bag_of_unigrams():
	print('unigrams')
	unigram_bow = []
	print('get unigrams start: ', time.time())
	unigrams = most_common_ngrams(source_text, 1, 500)
	print('get unigrams done: ', time.time())

	for unigram in unigrams:
		counts = [sum(unigram == n for n in make_ngrams(essay, 1))/len(essay) for essay in stemmed_text]
		unigram_bow.append(counts)
	print('unigrams done: ', time.time())

	return unigram_bow, unigrams

def log(fvec, hvec):
	with open('log.csv', 'a') as lfile:
		lwriter = csv.writer(lfile)
		lwriter.writerow(hvec)
		lwriter.writerows(fvec)

def extract_features(text, conf):
	all = False
	if len(conf)==0:
		all = True

	## glabal variables for text
	global stemmed_text
	global source_text
	global tokenized_text

	## Add pickling to save preprocessing time
	try:
		with open('stemmed_text.pkl', 'rb') as f1:
			stemmed_text = pickle.load(f1)
		with open('source_text.pkl','rb') as f2:
			source_text = pickle.load(f2)
		with open('tokenized_text.pkl','rb') as f3:
			tokenized_text = pickle.load(f3)
	except:
		source_text = text
		preprocess()			# we'll use global variables to pass the data around

	features = []		# features will be list of lists, each component list will have the same length as the list of input text
	header = []

	# extract requested features: FILL IN HERE

	if 'bag_of_function_words' in conf or all:
		fvec, hvec = bag_of_function_words()
		features.extend(fvec)
		header.extend(hvec)
		log(fvec, hvec)

	if 'bag_of_pos_trigrams' in conf or all:
		fvec, hvec = bag_of_pos_trigrams()
		features.extend(fvec)
		header.extend(hvec)
		log(fvec, hvec)

	if 'bag_of_pos_bigrams' in conf or all:
		fvec, hvec = bag_of_pos_bigrams()
		features.extend(fvec)
		header.extend(hvec)
		log(fvec, hvec)

	if 'bag_of_pos_unigrams' in conf or all:
		fvec, hvec = bag_of_pos_unigrams()
		features.extend(fvec)
		header.extend(hvec)
		log(fvec, hvec)

	if 'bag_of_trigrams' in conf or all:
		fvec, hvec = bag_of_trigrams()
		features.extend(fvec)
		header.extend(hvec)
		log(fvec, hvec)

	if 'bag_of_bigrams' in conf or all:
		fvec, hvec = bag_of_bigrams()
		features.extend(fvec)
		header.extend(hvec)
		log(fvec, hvec)

	if 'bag_of_unigrams' in conf or all:
		fvec, hvec = bag_of_unigrams()
		features.extend(fvec)
		header.extend(hvec)
		log(fvec, hvec)

	features = numpy.asarray(features).T.tolist() # transpose list of lists so its dimensions are #instances x #features

	with open('features.csv', 'w') as ffile:
		fwriter = csv.writer(ffile)
		fwriter.writerow(header)
		fwriter.writerows(features)

	return features
