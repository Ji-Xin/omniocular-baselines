import os

from util.preprocess import split_json_string


def vulas_diff_token():
    train_split = list()
    validation_split = list()
    test_split = list()
    with open(os.path.join(os.pardir, 'omniocular-data', 'datasets', 'vulas_diff_token', 'train.tsv')) as tsv_file:
        for line in tsv_file:
            repo, sha, code, label = line.split('\t')
            train_split.append((label, split_json_string(code)))
    with open(os.path.join(os.pardir, 'omniocular-data', 'datasets', 'vulas_diff_token', 'test.tsv')) as tsv_file:
        for line in tsv_file:
            repo, sha, code, label = line.split('\t')
            validation_split.append((label, split_json_string(code)))
    with open(os.path.join(os.pardir, 'omniocular-data', 'datasets', 'vulas_diff_token', 'test.tsv')) as tsv_file:
        for line in tsv_file:
            repo, sha, code, label = line.split('\t')
            test_split.append((label, split_json_string(code)))
    return train_split, validation_split, test_split
