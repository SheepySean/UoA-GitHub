from pymongo import MongoClient
from Database.Configuration import host, port

from NLP.keyword_extractor import KeywordExtractor
from NLP.latent_dirichlet_allocation import LatentDirichletAllocation
from NLP.latent_semantic_analysor import LatentSemanticAnalyser


def rankKeywords(docs):
    """
    Sorts the extracted keywords from docs by frequency and returns the result
    :param docs: documents
    :type docs: list
    :return: sorted words by frequency
    :rtype: list
    """
    kwpcsr = KeywordExtractor(docs)

    words_occurences = kwpcsr.count_keywords()


    #print('Occurences for each stemmed and lemmatized keyword: ')
    #print(words_occurences)
    #print('Sorted by descending occurence: ')
    sorted_words = sorted(words_occurences, key=words_occurences.__getitem__, reverse=True)
    #print(sorted_words)
    return sorted_words

def LDA(docs, topics, passes, save_filename):
    lda = LatentDirichletAllocation(docs, workers=3)
    model, corpus, dictionary = lda.compute(topics, passes)
    lda.save_to_file(save_filename, model, corpus, dictionary)
    return model, corpus, dictionary

def LSA(docs, topics, save_filename):
    lsi = LatentSemanticAnalyser(docs)
    return lsi.compute(topics, save_filename)

client = MongoClient(host, port)

db = client.uoa_gh

pull_request_comments = db.pull_request_comments.find({})

comments = []

for comment in pull_request_comments:
    comments.append(comment['body'])


print(rankKeywords(comments))
topics = 20
passes = 10
model, crp, dic = LDA(comments, topics, passes, 'lda_questions_{}_{}'.format(topics, passes))

for topic in model.show_topics():
    print(topic[0], topic[1])