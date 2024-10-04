import argparse
import numpy as np
import pandas as pd
import nltk
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import svm

DATA_PATH = './movie_reviews'
MAX_FEATURES = 3000
K_FOLDS = 3
MIN_DF = 2
MAX_DF = 0.7
NGRAM_RANGE = (1, 1)
# STRIP_ACCENTS = 'unicode'
# STOP_WORDS = 'english'
ANALYZER = 'char'


def main():
    # init arg parser
    parser = argparse.ArgumentParser(description="P2 Movie Sentiment Classifier CLI")
    parser.add_argument("model", choices=['nb', 'svm-bow', 'svm-tf'])
    args = parser.parse_args()

    # init model
    model = {
        'nb': MultinomialNB(),
        'svm-bow': svm.SVC(),
        'svm-tf': svm.SVC(),
    }[args.model]

    # load the data set
    movie_ds = load_files(DATA_PATH, shuffle=True)
    x = np.array(movie_ds.data)
    y = np.array(movie_ds.target)

    # init vectorizer and transformer
    count_vectorizer = CountVectorizer(
        min_df=MIN_DF,
        # max_df=MAX_DF,
        max_features=MAX_FEATURES,
        tokenizer=nltk.word_tokenize,
        token_pattern=None,
        ngram_range=NGRAM_RANGE,
        # strip_accents=STRIP_ACCENTS,
        # stop_words=STOP_WORDS,
    )

    tfidf_transformer = None
    if args.model != 'svm-bow':
        tfidf_transformer = TfidfTransformer()

    # run cross validation
    results = run_cv(model, x, y, count_vectorizer, tfidf_transformer)
    print(f"# model={args.model}, k_folds={K_FOLDS}, max_features={MAX_FEATURES}, min_df={MIN_DF}, max_df={MAX_DF}, "
          f"ngram_range={NGRAM_RANGE}\n{results}")


def run_cv(model, x, y, count_vectorizer, tfidf_transformer=None):
    results = []
    k_fold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=777)
    for train, test in k_fold.split(x, y):
        # split fold into training & testing sets
        x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]

        # fit & transform data sets
        x_train = count_vectorizer.fit_transform(x_train)
        x_test = count_vectorizer.transform(x_test)

        if tfidf_transformer:
            x_train = tfidf_transformer.fit_transform(x_train)
            x_test = tfidf_transformer.transform(x_test)

        # train the model
        model.fit(x_train, y_train)

        # test the model
        y_hat = model.predict(x_test)

        # evaluate the model
        tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
        results.append({
            'accuracy': accuracy_score(tp=tp, fp=fp, tn=tn, fn=fn),
            'recall': recall_score(tp=tp, fn=fn),
            'precision': precision_score(tp=tp, fp=fp),
            'f1': f1_score(tp=tp, fp=fp, fn=fn),
        })

    # analyze the run results
    results_df = pd.DataFrame.from_records(results).mean()

    return results_df


def f1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)


def precision_score(tp, fp):
    return tp / (tp + fp)


def accuracy_score(tp, fp, tn, fn):
    return (tp + tn) / (tp + fp + tn + fn)


def recall_score(tp, fn):
    return tp / (tp + fn)


if __name__ == "__main__":
    main()
