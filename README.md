# topic-co-occurrence

This is a simple end-to-end process to create a topic co-occurrence matrix. To run, change the data source in the code, and run the `topic_model.py` script. The csv file is expected to have a "content" column.

There are many ways to optimize this code. For example, you may want to lemmatize the tokens or use `nltk` to tokenize the text. Those parts are kept simple deliberately, not to distract from the main purpose.

The final output of the script are sentences formed from the topics of each document in the corpus. This specific format is used as input to keygraph analysis of the co-ocurring topics.


