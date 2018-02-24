import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from ML.classifier import Classifier
from NLP.input_preprocessor import InputPreprocessor
from Utils.logger import logger

from ML.classifier_data import ClassifierData


class C45DecisionTreeClassifier:
    """
    Class that uses TF-IDF vector text representation and the SGD algorithm to classify texts.
    """
    def __init__(self, data, target, target_names, preprocess=False):
        """
        Initializes the classifier by training it on the given data.
        :param data: text documents to train the classifier on
        :type data: list
        :param target: category indexes for each text document
        :type target: list
        :param target_names: category names
        :type target_names: list
        """

        self.__clf_data = ClassifierData(data, target, target_names)
        if preprocess:
            vect = CountVectorizer()
            analyzer = vect.build_analyzer()
            ipp = InputPreprocessor(None)
            def preprocess(doc):
                return [ipp.normalise(word) for word in analyzer(doc)]

            vectorizer = CountVectorizer(analyzer=preprocess)

        else:
            vectorizer = CountVectorizer()
        self.__text_clf = Pipeline([('vect', vectorizer),
                                    ('tfidf', TfidfTransformer()),
                                    ('clf', DecisionTreeClassifier(criterion='entropy')),
                                    ])


    @property
    def data(self):
        return self.__clf_data.data

    @property
    def target(self):
        return self.__clf_data.target

    @property
    def target_names(self):
        return self.__clf_data.target_names

    def train(self):
        self.__text_clf = self.__text_clf.fit(self.__clf_data.data, self.__clf_data.target)

    @property
    def text_clf(self):
        return self.__text_clf

    def evaluate_precision(self, n_splits):
        """
            Evaluates the precision of the classifier by running its prediction on the training data set
            :param n_splits: number of splits to split the training data into
            :type n_splits: int
            :return: the percentage of correctly predicted categories
            :rtype: float
        """
        if n_splits == 1:
            return np.mean(self.predict(self.__clf_data.data) == self.__clf_data.target)

        length = len(self.__clf_data.data)
        sub_length = int(length / n_splits)
        logger.info("Training data splitted in {} splits, each of length of {}, for a total length of {}".format(n_splits, sub_length, length))

        splits = []
        for i in range(0, n_splits):
            split = ClassifierData(self.__clf_data.data[i:i+sub_length], self.__clf_data.target[i:i+sub_length], self.__clf_data.target_names)
            splits.append(split)


        total = 0
        precisions = []

        for split in splits:
            training_data = ClassifierData([], [], self.target_names)
            total += len(split.data)

            for other_split in splits:
                if other_split != split:
                    for i in range(len(other_split.data)):
                        training_data.data.append(other_split.data[i])
                    for i in range(len(other_split.data)):
                        training_data.target.append(other_split.target[i])

            test_classifier = C45DecisionTreeClassifier(training_data.data, training_data.target, training_data.target_names)
            test_classifier.train()
            logger.debug("Trained on split {}".format(split))

            precision = np.mean(test_classifier.predict(split.data) == split.target)
            logger.debug("Precision on other split {} : {}".format(split, precision))

            precisions.append(precision)
            logger.debug("Total added splits length: {}".format(total))

        return np.mean(precisions)

    def predict(self, data):
        """
        Predicts the category of the data parameter.
        :param data: text document
        :type data: str
        :return: the predicted category index
        :rtype: int
        """
        return self.__text_clf.predict(data)







