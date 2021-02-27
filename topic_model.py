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

corpus = load_corpus("data/assertions.csv")
show_corpus(corpus)
