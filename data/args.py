from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Omniocular Baselines")
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--dataset', type=str, default='VulasDiffToken', choices=['VulasDiffToken'])

    args = parser.parse_args()
    return args