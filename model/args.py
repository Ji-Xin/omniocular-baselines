from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Chronicle")
    parser.add_argument('--model', type=str, default='LogisticRegression', choices=['LogisticRegression', 'LinearSVC', 'DecisionTree'])
    parser.add_argument('--dataset', type=str, default='VulasDiffToken', choices=['VulasDiffToken'])

    args = parser.parse_args()
    return args