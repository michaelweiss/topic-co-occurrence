# -*- coding: utf-8 -*-
# simple end-to-end process to create topic co-occurrence graph

import pandas as pd
from topics import TopicModel

tm = TopicModel()

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

corpus = load_corpus("data/assertions.csv")
show_corpus(corpus)

stopwords = read_stopwords("stopwords-en.txt")

tokens = [[w for w in tokenize(document) if w not in stopwords and w.isalnum()]
			for document in corpus['content']]
print(tokens)