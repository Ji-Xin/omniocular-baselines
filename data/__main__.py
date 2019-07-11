from data import summary
from data import load
from data.args import get_args

if __name__ == '__main__':
    args = get_args()

    if args.dataset == 'VulasDiffToken':
        train_split, validation_split, test_split = load.vulas_diff_token()
    else:
        raise Exception("Unsupported dataset")

    data_x = [x[1] for x in train_split]
    data_x.extend([x[1] for x in test_split])
    data_x.extend([x[1] for x in validation_split])

    data_y = [x[0] for x in train_split]
    data_y.extend([x[0] for x in test_split])
    data_y.extend([x[0] for x in validation_split])

    print("Number of samples:", summary.num_samples(data_x))
    print("Label frequencies:", summary.label_freq(data_y))
    print("Average number of tokens in a sample:", summary.avg_num_tokens(data_x))
