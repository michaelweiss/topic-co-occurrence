# -*- coding: utf-8 -*-
# simple end-to-end process to create topic co-occurrence matrix

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

def document_topics_matrix(tokens, dictionary):
	return [lda.get_document_topics(bow) for bow in tokens_to_bow(tokens, dictionary)]

def show_document_topics_matrix(dtm):
	for i, topics in enumerate(dtm):
		print(i, topics)

def topic_co_occurrence_matrix(dtm, min_weight=0.1):
	return [[t for t, w in topics if w >= min_weight] for topics in dtm]

def tcom_to_sentences(tcom):
	for tco in tcom:
		tco = ["T{}".format(t) for t in tco]
		tco.append('.')
		print(" ".join(tco))

#-----------Main----------------
if __name__ == "__main__":
	corpus = load_corpus("data/abstracts.csv")
	# show_corpus(corpus)
	tokens = corpus_to_tokens(corpus)
	dictionary = Dictionary(tokens)
	lda = fit_lda(tokens, 10, dictionary)
	# show_topics(lda)
	dtm = document_topics_matrix(tokens, dictionary)
	# show_document_topics_matrix(dtm)
	tcom = topic_co_occurrence_matrix(dtm, 0.1)
	# for i, tco in enumerate(tcom):
	# 	print(i, tco)
	tcom_to_sentences(tcom)


