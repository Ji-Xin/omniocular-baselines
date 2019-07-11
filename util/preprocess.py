import json


def split_string(string):
    return string.split()


def split_json_string(string, max_length=8000):
    split_val = json.loads(string)
    return ' '.join(split_val[:min(max_length, len(split_val))])

