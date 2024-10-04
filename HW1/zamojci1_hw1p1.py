import nltk
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.model_selection import KFold
import argparse


PARAMETERS_PATH_WORDS = "zamojci1_params_pwords.pkl"
PARAMETERS_PATH_CLASS = "zamojci1_params_pclass.pkl"
FOLD_CHOICES = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
K_FOLDS = 3


def main():
    # init arg parser
    parser = argparse.ArgumentParser(description="NB Classifier CLI")
    parser.add_argument("mode", choices=['train', 'test', 'all'])
    parser.add_argument("fold_a",  nargs='?', choices=FOLD_CHOICES)
    parser.add_argument("fold_b", nargs='?', choices=FOLD_CHOICES)
    args = parser.parse_args()

    # init program mode
    mode = args.mode

    # run program
    if mode == "train":
        print(f"# Training folds: {args.fold_a}, {args.fold_b}")
        files = parse_fold(args.fold_a) + parse_fold(args.fold_b)
        if not files:
            print("Error: Loaded fold files are null.")
            return
        train(files)
        print(f"\tFinished training model parameters: {PARAMETERS_PATH_WORDS}, {PARAMETERS_PATH_CLASS}")
    elif mode == "test":
        print(f"# Testing fold: {args.fold_a}")
        files = parse_fold(args.fold_a)
        if not files:
            print("Error: Loaded fold files are null.")
            return
        test(files)
    else:
        cross_validate()


def cross_validate():
    print(f"# Running cross-validation: k_folds={K_FOLDS}")
    k_fold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=777)
    scores = []
    for train_idx, test_idx in k_fold.split(FOLD_CHOICES):
        print(f"\ttrain_folds={train_idx}, test_folds={test_idx}")
        train_files = flatten([parse_fold(FOLD_CHOICES[idx]) for idx in train_idx])
        test_files = flatten([parse_fold(FOLD_CHOICES[idx]) for idx in test_idx])

        train(train_files)
        scores.append(test(test_files))

    # analyze the run results
    scores_df = pd.DataFrame.from_records(scores).mean()
    print(f"# Average scores:\n{scores_df}")

    return scores_df


def train(_files):
    cls_freq = {}
    cls_prob = {}
    for category in movie_reviews.categories():
        # filter words for class & fold
        t_files = [f for f in _files if f in movie_reviews.fileids(categories=category)]
        t_words = movie_reviews.words(fileids=t_files)
        t_freq = nltk.FreqDist(t_words)

        # calculate multiclass data
        cls_freq[category] = dict(t_freq)
        cls_prob[category] = len(t_files) / len(_files)

    # probability of word given class - p(w | c)
    freq_df = pd.DataFrame.from_dict(cls_freq, orient='columns').fillna(0.0)
    p_words = (freq_df + 1) / (freq_df + 1).sum()
    p_words.to_pickle(PARAMETERS_PATH_WORDS)

    # probability of class - p(c)
    p_class = pd.DataFrame.from_dict(cls_prob, orient='index', columns=['p_class'])
    p_class.to_pickle(PARAMETERS_PATH_CLASS)

    return p_words, p_class


def test(_files):
    # load trained parameters
    p_words = pd.read_pickle(PARAMETERS_PATH_WORDS)
    p_class = pd.read_pickle(PARAMETERS_PATH_CLASS)
    logp_class = np.log(p_class['p_class'])

    # predict class for each test document
    f_results = []
    p_min = p_words.min().min()
    for f in _files:
        # load file words
        f_words = movie_reviews.words(fileids=f)

        # calculate probability of sentence given class - p(s | c)
        p_sentence = np.log(p_words.reindex(f_words, fill_value=p_min)).sum()

        # predict document class from max probability
        y_hat = logp_class.index[np.argmax(logp_class + p_sentence)]

        # append results
        f_results.append({
            'file': f,
            'y_hat': y_hat,
            'y_true': movie_reviews.categories(fileids=f).pop(),
        })

    # evaluate class prediction results
    results_df = pd.DataFrame.from_records(f_results, index='file')

    # confusion matrix
    tp = len(results_df.loc[(results_df['y_hat'] == results_df['y_true']) & (results_df['y_hat'] == 'pos')])
    fp = len(results_df.loc[(results_df['y_hat'] != results_df['y_true']) & (results_df['y_hat'] == 'pos')])
    tn = len(results_df.loc[(results_df['y_hat'] == results_df['y_true']) & (results_df['y_hat'] == 'neg')])
    fn = len(results_df.loc[(results_df['y_hat'] != results_df['y_true']) & (results_df['y_hat'] == 'neg')])

    # evaluation metrics
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # accuracy = (tp + tn) / (tp + fp + tn + fn)
    # f1 = (2 * precision * recall) / (precision + recall)
    scores = {
        'accuracy': accuracy_score(tp=tp, fp=fp, tn=tn, fn=fn),
        'recall': recall_score(tp=tp, fn=fn),
        'precision': precision_score(tp=tp, fp=fp),
        'f1': f1_score(tp=tp, fp=fp, fn=fn),
    }
    print(f"\tPrecision: {scores['precision']}, Recall: {scores['recall']}, F1: {scores['f1']}, Accuracy: {scores['accuracy']}")

    return scores


def flatten(xss):
    return [x for xs in xss for x in xs]


def f1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)


def precision_score(tp, fp):
    return tp / (tp + fp)


def accuracy_score(tp, fp, tn, fn):
    return (tp + tn) / (tp + fp + tn + fn)


def recall_score(tp, fn):
    return tp / (tp + fn)


def parse_fold(fold: str) -> [str]:
    if not fold:
        return []

    # calculate fold file starting index
    _, fold_idx = fold.split('fold')
    file_idx = (int(fold_idx) - 1) * 100

    # concatenate fold files from both categories
    fold_neg = movie_reviews.fileids(categories='neg')[file_idx:file_idx + 100]
    fold_pos = movie_reviews.fileids(categories='pos')[file_idx:file_idx + 100]
    fold_files = fold_pos + fold_neg

    return fold_files


if __name__ == "__main__":
    main()
