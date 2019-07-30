import numpy as np

from model.args import get_args
from data import load
from model import linear_svc, logistic_regression, decision_tree, adaboost

if __name__ == '__main__':
    args = get_args()
    random_state = 3454

    accuracy_values = list()
    precision_values = list()
    recall_values = list()
    f1_values = list()

    if args.dataset == 'VulasDiffToken':
        train_split, validation_split, test_split = load.vulas_diff_token()
    else:
        raise Exception("Unsupported dataset")

    train_x = [x[1] for x in train_split]
    train_y = np.array([[int(x0) for x0 in x[0].strip()] for x in train_split])
    test_x = [x[1] for x in test_split]
    test_y = np.array([[int(x0) for x0 in x[0].strip()] for x in test_split])
    validation_x = [x[1] for x in validation_split]
    validation_y = np.array([[int(x0) for x0 in x[0].strip()] for x in validation_split])

    train_y = [np.where(r == 1)[0][0] for r in train_y]
    test_y = [np.where(r == 1)[0][0] for r in test_y]
    validation_y = [np.where(r == 1)[0][0] for r in validation_y]

    if args.model == 'LogisticRegression':
        model = logistic_regression
    elif args.model == 'LinearSVC':
        model = linear_svc
    elif args.model == 'DecisionTree':
        model = decision_tree
    elif args.model == 'AdaBoost':
        model = adaboost
    else:
        raise Exception("Unsupported model")

    classifier, vectorizer = model.train(train_x, train_y, single_label=True, random_state=random_state)
    print("Dev:", model.evaluate(classifier, vectorizer, validation_x, validation_y))
    print("Test:", model.evaluate(classifier, vectorizer, test_x, test_y))
