# topic-co-occurrence

This is a simple end-to-end script to create a topic co-occurrence matrix. Change the data source in the code, and run `topic_model.py`. The csv file is expected to have a "content" column.

There are many ways to optimize this code. For example, you may want to lemmatize the tokens or use `nltk` to tokenize the text. Those parts are kept simple deliberately to focus on the main purpose.

The final output of the script are sentences formed from the topics of each document in the corpus. This specific format is used as input to [keygraph](https://github.com/michaelweiss/keygraph) analysis of the co-ocurring topics.