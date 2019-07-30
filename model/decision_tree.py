import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import graphviz

import util.preprocess


def train(train_x, train_y, single_label=True, random_state=37):
    np.random.seed(random_state)
    vectorizer = TfidfVectorizer(tokenizer=util.preprocess.split_string,
                                 min_df=0.01,
                                 strip_accents='ascii',
                                 lowercase=True)

    train_x = vectorizer.fit_transform(train_x)
    if single_label:
        classifier = DecisionTreeClassifier(criterion="gini", random_state=random_state)
    else:
        classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=random_state))
    classifier.fit(train_x, train_y)
    return classifier, vectorizer


def predict(classifier, vectorizer, predict_x):
    predict_x = vectorizer.transform(predict_x)
    return classifier.predict(predict_x)


def evaluate(classifier, vectorizer, evaluate_x, evaluate_y):
    evaluate_x = vectorizer.transform(evaluate_x)
    predict_y = classifier.predict(evaluate_x)
    accuracy = accuracy_score(evaluate_y, predict_y)
    precision = precision_score(evaluate_y, predict_y, average=None)[0]
    recall = recall_score(evaluate_y, predict_y, average=None)[0]
    f1 = f1_score(evaluate_y, predict_y, average=None)[0]
    return ('Accuracy:', accuracy,
            'Precision:', precision,
            'Recall:', recall,
            'F-score:', f1)


def predict_probabilities(classifier, vectorizer, predict_x):
    predict_x = vectorizer.transform(predict_x)
    return classifier.predict_proba(predict_x)


def visualize(classifier, vectorizer):
    feature_names = [x.replace('"', '\\"') for x in vectorizer.get_feature_names()]
    dot_data = export_graphviz(classifier, out_file=None,
                               feature_names=feature_names,
                               filled=True, rounded=True,
                               special_characters=False)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")
