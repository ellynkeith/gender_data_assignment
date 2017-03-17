# extract features from list of text instances based on configuration set of features

import nltk
import numpy as np
import re
import csv
from nltk.util import ngrams
import time
import gensim
import statistics

# print("loading model ", time.time())
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True) 

source_text = []
stemmed_text = []
pos_tags = []

def preprocess():
    print("preprocessing ", time.time())
    # first stem and lowercase words, then remove rare
    # lowercase
    global source_text
    source_text = [text.lower() for text in source_text]

    # tokenize
    print("tokenizing ", time.time())
    global tokenized_text
    tokenized_text = [nltk.word_tokenize(text) for text in source_text]

    # pos tags in preprocessing step
    print("pos tagging ", time.time())
    global pos_tags
    pos_tags = [[pos[1] for pos in nltk.pos_tag(text)] for text in tokenized_text]

    global lex_pos_tags
    lex_pos_tags = [nltk.pos_tag(text) for text in tokenized_text]

    # stem
    print("stemming ", time.time())
    porter = nltk.PorterStemmer()
    global stemmed_text
    stemmed_text = [[porter.stem(t) for t in tokens] for tokens in tokenized_text]

    # remove rare
    print("removing rare ", time.time())
    vocab = nltk.FreqDist(w for line in stemmed_text for w in line)
    rarewords_list = set(vocab.hapaxes())
    stemmed_text = [['<RARE>' if w in rarewords_list else w for w in line] for line in stemmed_text]
    # note that source_text will be lowercased, but only stemmed_text will have rare words removed

def bag_of_function_words():
    print("bag of function words ", time.time())
    bow = []
    for sw in nltk.corpus.stopwords.words('english'):
        counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, line)) for line in source_text]
        bow.append(counts)
    print('bow stuff: ',len(bow), len(bow[0]))
    return bow, nltk.corpus.stopwords.words('english')

# FILL IN OTHER FEATURE EXTRACTORS

def make_ngrams(text, nGramCount):
    '''
    param text: list w/ sublists, sublists are tokens
    param nGramCount: integer indicating ngram to generate
    return: dictionary of ngrams, sorted by frequency
    '''
    ngramCounts = nltk.FreqDist(nltk.ngrams([word for doc in text for word in doc], nGramCount))
    return ngramCounts

def most_common(text, n, ngramCount):
    '''
    param text: list w/sublists, sublists are tokens
    param n: integer indicating how many of most common ngrams to return
    param ngramCount: integer indication ngram to generate
    return: list of n most common ngrams, w/o counts
    '''

    ngramFreqDist = make_ngrams(text, ngramCount)

    # most_common_ngrams =

    return [x[0] for x in ngramFreqDist.most_common(n)]

def replace_open_class(pos_tagged_sent):
    mixed_sent = [word[1] if word[0] not in nltk.corpus.stopwords.words("english") else word[0] for word in pos_tagged_sent]
    return mixed_sent

#mixed = [replace_open_class(nltk.pos_tag(sent)) for sent in tokenized_text]

## ngram features
## Added character level trigrams and bigrams
## added lex-pos ngrams
def bag_of_lex_pos_trigrams():
    print("lex/pos trgrams ",time.time())
    top_500_trigrams = most_common(replace_open_class(lex_pos_tags), 500,3)
    print("top 10 lex/pos trigrams ", top_500_trigrams[:10])
    essay_fds = [nltk.FreqDist(ngrams(essay, 3)) for essay in replace_open_class(lex_pos_tags)]
    bigram_counts = [[essay_fd[bg] for bg in top_500_trigrams] for essay_fd in essay_fds]
    return np.asarray(bigram_counts).T.tolist(), "top_500_lex_pos_trigrams"

def bag_of_lex_pos_bigrams():
    print("lex/pos bigrams ",time.time())
    top_100_bigrams = most_common(replace_open_class(lex_pos_tags), 100,2)
    print("top 10 lex/pos bugrams ", top_100_bigrams[:10])
    essay_fds = [nltk.FreqDist(ngrams(essay, 2)) for essay in replace_open_class(lex_pos_tags)]
    bigram_counts = [[essay_fd[bg] for bg in top_100_bigrams] for essay_fd in essay_fds]
    return np.asarray(bigram_counts).T.tolist(), "top_100_lex_pos_bigrams"


def bag_of_character_trigrams():
    print("bag of character trigrams ", time.time())
    top_500_trigrams = most_common(source_text, 500, 3)
    print('top char trigram ', top_500_trigrams[0])

    print('making essay freq dists ', time.time())
    essay_fds = [nltk.FreqDist(ngrams(essay, 3)) for essay in source_text]
    print('counting trigrams ', time.time())
    trigram_counts = [[essay_fd[tg] for tg in top_500_trigrams] for essay_fd in essay_fds]
#	trigram_counts = [[tg for essay_fd[tg] in essay_fds] for trigram in top_500_trigrams]
    print('trigram: ',len(trigram_counts), len(trigram_counts[0]))
    return np.asarray(trigram_counts).T.tolist(), "top_500_char_trigrams"

def bag_of_pos_trigrams():
    print("bag of pos trigrams ", time.time())
    top_500_trigrams = most_common(pos_tags, 500, 3)
    print('top pos trigram ', top_500_trigrams[0])

    print('making essay freq dists ', time.time())
    essay_fds = [nltk.FreqDist(ngrams(essay, 3)) for essay in pos_tags]
    print('counting trigrams ', time.time())
    trigram_counts = [[essay_fd[tg] for tg in top_500_trigrams] for essay_fd in essay_fds]
#	trigram_counts = [[tg for essay_fd[tg] in essay_fds] for trigram in top_500_trigrams]
    print('trigram: ',len(trigram_counts), len(trigram_counts[0]))
    return np.asarray(trigram_counts).T.tolist(), "top_500_pos_trigrams"

def bag_of_character_bigrams():
    print("bag of char bigrams ", time.time())
    top_100_bigrams = most_common(source_text, 100, 2)
    print('top char bigram ', top_100_bigrams[0])

    print('making essay freq dists ', time.time())
    essay_fds = [nltk.FreqDist(ngrams(essay, 2)) for essay in source_text]
    print('counting bigrams ', time.time())
    bigram_counts = [[essay_fd[bg] for bg in top_100_bigrams] for essay_fd in essay_fds]
    return np.asarray(bigram_counts).T.tolist(), "top_100_char_bigrams"

def bag_of_pos_bigrams():
    print("bag of pos bigrams ", time.time())
    top_100_bigrams = most_common(pos_tags, 100, 2)
    print('top pos bigram ', top_100_bigrams[0])

    print('making essay freq dists ', time.time())
    essay_fds = [nltk.FreqDist(ngrams(essay, 2)) for essay in pos_tags]
    print('counting bigrams ', time.time())
    bigram_counts = [[essay_fd[bg] for bg in top_100_bigrams] for essay_fd in essay_fds]
    return np.asarray(bigram_counts).T.tolist(), "top_100_pos_bigrams"

def bag_of_pos_unigrams():
    print("bag of pos unigrams ", time.time())
    bag_of_pos_unigrams = make_ngrams(pos_tags, 1)
    print('here is a pos unigram ', bag_of_pos_unigrams[0])

    essay_fds = [nltk.FreqDist(ngrams(essay, 1)) for essay in pos_tags]
    print('counting unigrams ', time.time())
    unigram_counts = [[essay_fd[ng] for ng in bag_of_pos_unigrams] for essay_fd in essay_fds]
    return np.asarray(unigram_counts).T.tolist(), "bag_of_pos_unigrams"

def bag_of_trigrams():
    print("bag of trigrams ", time.time())
    top_500_trigrams = most_common(stemmed_text, 500, 3)
    print('top trigram ', top_500_trigrams[0])

    print('making essay freq dists ', time.time())
    essay_fds = [nltk.FreqDist(ngrams(essay, 3)) for essay in stemmed_text]
    print('counting trigrams ', time.time())
    trigram_counts = [[essay_fd[tg] for tg in top_500_trigrams] for essay_fd in essay_fds]
    return np.asarray(trigram_counts).T.tolist(), "top_500_trigrams"

def bag_of_bigrams():
    print("bag of bigrams ", time.time())
    top_100_bigrams = most_common(stemmed_text, 100, 2)
    print('top bigram ', top_100_bigrams[0])

    print('making essay freq dists ', time.time())
    essay_fds = [nltk.FreqDist(ngrams(essay, 2)) for essay in stemmed_text]
    print('counting bigrams ', time.time())
    bigram_counts = [[essay_fd[bg] for bg in top_100_bigrams] for essay_fd in essay_fds]
    return np.asarray(bigram_counts).T.tolist(), "top_100_bigrams"

def bag_of_unigrams():
    print("bag of unigrams ", time.time())
    bag_of_unigrams = make_ngrams(stemmed_text, 1)
    print('and a unigram ', bag_of_unigrams[0])
    essay_fds = [nltk.FreqDist(ngrams(essay, 1)) for essay in stemmed_text]
    print('counting unigrams ', time.time())
    unigram_counts = [[essay_fd[ng] for ng in bag_of_unigrams] for essay_fd in essay_fds]
    return np.asarray(unigram_counts).T.tolist(), "bag_of_unigrams"

## complexity features
def characters_per_word():
    print("chars per word: ", time.time())
    char_avgs = []
    ## generate list of word lengths for each essay, return mean length
    for essay in tokenized_text:
        avg = statistics.mean([len(word) for word in essay])
        char_avgs.append(avg)
    return char_avgs, "char_avgs"

def unique_words_ratio():
    print("unique words ratio: ",time.time())
    unique_word_ratio = []

    for essay in tokenized_text:
        unique = len(set(essay))/len(essay)
        unique_word_ratio.append(unique)
    return unique_word_ratio, "unique_word_ratio"

def words_per_sentence():
    print("words per sentence: ",time.time())
    word_avgs = []
    essays = [nltk.sent_tokenize(essay) for essay in source_text]
    for essay in essays:
        avg = statistics.mean([len(line) for line in essay])
        word_avgs.append(avg)
    return word_avgs, "word_avgs"

## word vector feature
def makeAvgWordVec(doc, model):
    '''
    param doc: string of text
    param model: word vectorization model
    return: average word vector for given string
    '''
    featureVec = np.zeros((1,300), dtype="float32")
    vec = []
    ## append each word vector to list
    for word in doc:
        if word in model.vocab:
            vec.append(model[word])

    ## convert list to array, return mean of rows
    featureVec = np.array(vec).mean(axis = 0)

    return featureVec

def avg_feature_vecs(text, model):
    '''
    param text: list of strings
    param model: word vectorization model
    return: list of averaged word vectors
    '''
    print("avg feature vecs", time.time())
    ## remove stopwords
    text = [[word for word in line if word not in nltk.corpus.stopwords.words('english')] for line in text]
    textFeatureVecs = np.zeros((len(text), 300), dtype="float32")
    for i in range(len(text)):
        textFeatureVecs[i] = makeAvgWordVec(text[i], model)

    return np.asarray(textFeatureVecs).T.tolist(), "avgFeatureVecs"


def log(fvec, hvec):
    with open('log.csv', 'a') as lfile:
        lwriter = csv.writer(lfile)
        lwriter.writerow(hvec)
        lwriter.writerows(fvec)

def extract_features(text, conf):
    print("extracting features", time.time())
    all = False
    if len(conf)==0:
        all = True

    global source_text
    source_text = text			# we'll use global variables to pass the data around
    preprocess()

    features = []		# features will be list of lists, each component list will have the same length as the list of input text
    header = []

    #extract requested features: FILL IN HERE
    if 'bag_of_function_words' in conf or all:
    	fvec, hvec = bag_of_function_words()
    	features.extend(fvec)
    	header.extend(hvec)
    	log(fvec, hvec)


    if "bag_of_lex_pos_bigrams" in conf or all:
        fvec, hvec = bag_of_lex_pos_bigrams()
        features.extend(fvec)
        header.extend(hvec)
        log(fvec,hvec)

    if "bag_of_lex_pos_trigrams" in conf or all:
        fvec, hvec = bag_of_lex_pos_trigrams()
        features.extend(fvec)
        header.extend(hvec)
        log(fvec, hvec)

    if 'bag_of_character_trigrams' in conf or all:
    	fvec, hvec = bag_of_character_trigrams()
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

    if 'bag_of_character_bigrams' in conf or all:
        fvec, hvec = bag_of_character_bigrams()
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

    # if 'bag_of_unigrams' in conf or all:
    # 	fvec, hvec = bag_of_unigrams()
    # 	features.extend(fvec)
    # 	header.extend(hvec)
    # 	log(fvec, hvec)
    #
    # if 'characters_per_word' in conf or all:
    # 	fvec, hvec = characters_per_word()
    # 	features.extend(fvec)
    # 	header.extend(hvec)
    # 	log(fvec, hvec)
    #
    # if 'unique_words_ratio' in conf or all:
    # 	fvec, hvec = unique_words_ratio()
    # 	features.extend(fvec)
    # 	header.extend(hvec)
    # 	log(fvec, hvec)
    #
    # if 'words_per_sentence' in conf or all:
    # 	fvec, hvec = words_per_sentence()
    # 	features.extend(fvec)
    # 	header.extend(hvec)
    # 	log(fvec, hvec)

    # if 'avg_feature_vecs' in conf or all:
    # 	fvec, hvec = avg_feature_vecs(tokenized_text, model)
    # 	features.extend(fvec)
    # 	header.extend(hvec)
    # 	log(fvec, hvec)

    features = np.asarray(features).T.tolist() # transpose list of lists so its dimensions are #instances x #features
    ## something weird and inefficient is happening here--vectors are getting transposed when returned from function, and then again here--redundant?

    with open('features.csv', 'w') as ffile:
        fwriter = csv.writer(ffile)
        fwriter.writerow(header)
        fwriter.writerows(features)

    return features