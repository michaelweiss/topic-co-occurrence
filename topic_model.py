# -*- coding: utf-8 -*-
# simple end-to-end process to create topic co-occurrence graph

import pandas as pd

from gensim.models import LdaModel
from gensim.corpora import Dictionary

def load_corpus(file):
	documents = pd.read_csv(file)
	return documents

def show_corpus(corpus):
	for i, document in enumerate(corpus['content']):
		print(i, document)

def read_stopwords(file):
	file = open(file, 'r')
	return [w.strip() for w in file.read().split('\n')]

def tokenize(document):
	return [w.lower() for w in document.split()]

def corpus_to_tokens(corpus):
	stopwords = read_stopwords("stopwords-en.txt")
	return [[w for w in tokenize(document) if w not in stopwords and w.isalnum()]
		for document in corpus['content']]

def tokens_to_bow(tokens, dictionary):
	return [dictionary.doc2bow(document) for document in tokens]

def fit_lda(tokens, number_of_topics, dictionary):
	return LdaModel(tokens_to_bow(tokens, dictionary), number_of_topics, dictionary)

def show_topics(lda):
	for topics in lda.show_topics():
		print(topics)

def ids_to_words(bow, dictionary):
	return [(dictionary.id2token[w], f) for w, f in bow]

#-----------Main----------------
if __name__ == "__main__":
	corpus = load_corpus("data/assertions.csv")
	# show_corpus(corpus)
	tokens = corpus_to_tokens(corpus)
	dictionary = Dictionary(tokens)
	lda = fit_lda(tokens, 10, dictionary)
	show_topics(lda)

	for bow in tokens_to_bow(tokens, dictionary):
		print(ids_to_words(bow, dictionary))
		topics = lda.get_document_topics(bow)
		print(topics)



