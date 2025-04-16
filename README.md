# CSE842: Natural Language Processing Coursework

This repository contains my coursework for CSE842, a graduate-level Natural Language Processing (NLP) course completed as part of my Master’s in Computer Science and Engineering. It includes three projects/homeworks, demonstrating my proficiency in designing and implementing NLP models for sentiment analysis, part-of-speech (POS) tagging, text classification, and transformer-based fine-tuning. The projects emphasize NLTK, scikit-learn, Pandas, PyTorch, Keras, and HuggingFace Transformers, showcasing my readiness for machine learning and NLP engineering roles.

# Table of Contents

- [CSE842: Natural Language Processing Coursework](#cse842-natural-language-processing-coursework)
  - [Projects](#projects)
    - [Homework 1: Sentiment Analysis with Naïve Bayes and SVM](#homework-1-sentiment-analysis-with-naïve-bayes-and-svm)
    - [Homework 2: Hidden Markov Model and Neural POS Tagging](#homework-2-hidden-markov-model-and-neural-pos-tagging)
    - [Homework 3: BERT Fine-Tuning and Non-Neural Text Classification](#homework-3-bert-fine-tuning-and-non-neural-text-classification)
  - [Skills Demonstrated](#skills-demonstrated)

## Projects

### Homework 1: Sentiment Analysis with Naïve Bayes and SVM

#### Description
Implemented and evaluated sentiment analysis models on the Movie Reviews Sentiment Polarity Dataset v2.0 (Pang, Lee, Vaithyanathan), predicting positive or negative sentiment. The project included two parts: a custom Naïve Bayes classifier from scratch and scikit-learn-based Naïve Bayes and SVM models using different feature sets.

#### Approach
- **Problem 1: Custom Naïve Bayes Classifier**:
  - Built a Naïve Bayes classifier using unigrams as features, processing punctuation as words.
  - Implemented 3-fold cross-validation, training on two folds and testing on the third per run.
  - Created a vocabulary from training data, computing `P(w|c)` and `P(c)` with Laplace smoothing (k=1) in log space.
  - Trained using CLI command `train fold1 fold2`, saving model parameters (word and class probabilities) as pickled Pandas DataFrames (`zamojci1_params_pclass/pwords.pkl`).
  - Tested with `test fold3`, predicting document sentiment by summing log probabilities and comparing to ground truth.
  - Evaluated precision, recall, F1, and accuracy, averaging metrics across folds.
- **Problem 2: NLTK & scikit-learn Models**:
  - Implemented three models: Naïve Bayes, SVM with bag-of-words (BoW), and SVM with TF-IDF features.
  - Used NLTK to load the dataset and scikit-learn’s `CountVectorizer` (max_features=3000, min_df=2) for feature extraction, with 3-fold cross-validation.
  - For SVM-TF-IDF, applied `TfidfTransformer` to BoW features to weigh word importance.
  - Unified pipeline in `zamojci1_hw1p2_all.py`, allowing model selection via CLI argument (`nb`, `svm-bow`, `svm-tf`).
  - Evaluated precision, recall, F1, and accuracy, averaging across folds.
- **Extra Work**:
  - Experimented with SVM-TF-IDF parameters:
    - Removed high-frequency words (`max_df=0.7`), improving performance.
    - Tested bigrams (`ngram_range=(2,2)`) and unigram+bigrams (`(1,2)`), finding unigrams optimal.
    - Tried character n-grams (`analyzer=char`), with no performance change.
    - Set `max_features=None` and `min_df=1` with `max_df=0.7`, slightly improving recall but lowering overall metrics.

#### Tools
- **NLTK**: Loaded and processed the Movie Reviews dataset, extracted word frequencies.
- **scikit-learn**: Implemented Naïve Bayes and SVM models, used `CountVectorizer` and `TfidfTransformer` for feature extraction.
- **Pandas**: Stored and managed model parameters (probabilities) as DataFrames.
- **Python**: Built custom Naïve Bayes with CLI support, leveraging log-space calculations.

#### Results
- **Custom Naïve Bayes** (3-fold CV):
  - Accuracy: 81.11%
  - Precision: 82.16%
  - Recall: 79.56%
  - F1: 80.83%
- **scikit-learn Models** (3-fold CV, max_features=3000, min_df=2):
  - **Naïve Bayes**: Accuracy: 79.10%, Precision: 82.42%, Recall: 74.13%, F1: 78.00%
  - **SVM-BoW**: Accuracy: 71.40%, Precision: 76.03%, Recall: 62.51%, F1: 68.59%
  - **SVM-TF-IDF**: Accuracy: 82.70%, Precision: 82.17%, Recall: 83.60%, F1: 82.85%
- **Extra Experiments**:
  - SVM-TF-IDF with `max_df=0.7` outperformed baseline, reducing noise from frequent words.
  - Bigrams and character n-grams underperformed, confirming unigrams as optimal for this dataset.
- **Output**: Saved model parameters (`zamojci1_params_pclass/pwords.pkl`) and evaluation metrics in report, with code files (`zamojci1_hw1p1.py`, `zamojci1_hw1p2_all.py`).

![image](https://github.com/user-attachments/assets/82b52f00-419e-489d-bcee-8fd7bdede60a)

#### Key Skills
- Custom NLP model development (Naïve Bayes).
- Feature engineering (BoW, TF-IDF).
- Cross-validation and evaluation metrics.
- CLI-based pipeline implementation.
- Parameter tuning and experimentation.

### Homework 2: Hidden Markov Model and Neural POS Tagging

#### Description
Developed models for part-of-speech (POS) tagging using the Brown and Treebank corpora with the universal tagset (12 tags). The project included two parts: calculating Hidden Markov Model (HMM) probability matrices and implementing a neural network (RNN/LSTM) for POS tagging, plus extra work with Keras.

#### Approach
- **Problem 1: Hidden Markov Model**:
  - Calculated maximum likelihood estimates (MLE) for HMM transition (`A`) and emission (`B`) matrices using the Brown corpus (news category).
  - For the transition matrix, used NLTK’s `bigrams` to count tag transitions (`t_i-1` to `t_i`), computing `P(t_i|t_i-1)` via conditional frequency distributions, converted to a Pandas DataFrame.
  - For the emission matrix, counted word-tag pairs (lowercased words), computing `P(w|t)` for words `science`, `all`, `well`, `like`, `but`, `blue`, `city`, and all tags.
  - Output matrices as DataFrames, with `A` showing all 12x12 tag transitions and `B` focusing on specified words.
- **Problem 2: Neural Network (RNN)**:
  - Implemented an RNN in PyTorch for POS tagging on the Treebank corpus, using an 80-20 train-test split with 5-fold cross-validation.
  - Engineered features: word2vec embeddings (100D, window=5, min_count=1), plus binary features (is_alpha, is_digit, is_lowercase, is_titlecase, is_uppercase), yielding 106D vectors.
  - Padded sentences to 271 words (max length), normalized input tensors.
  - Tested hyperparameters: criterion (NLLLoss, CrossEntropyLoss), optimizer (RMSprop, Adam), hidden nodes (64, 128), learning rate (0.001, 0.0001), across 16 configurations (100 epochs).
  - Evaluated accuracy by averaging best fold scores.
- **Extra Work**:
  - Re-implemented POS tagging with Keras, using LSTM models (return_sequences=True) for many-to-many predictions.
  - Preprocessed data by mapping words to frequency-based indices (vocab size ~27,000), padding to 200 words.
  - Tested hyperparameters: optimizer (RMSprop, Adam), embedding dimension (32, 64, 128), LSTM hidden nodes (25, 50, 100), across 18 configurations (3 epochs, 5-fold CV).
  - Used sparse categorical cross-entropy loss, adding a dense layer (12 outputs).

#### Tools
- **NLTK**: Loaded Brown and Treebank corpora, extracted bigrams and frequency distributions.
- **PyTorch**: Built RNN models with word2vec embeddings and binary features.
- **Keras**: Implemented LSTM models with frequency-based embeddings.
- **Pandas**: Stored HMM matrices and processed data.
- **scikit-learn**: Supported cross-validation and vectorization.
- **Gensim**: Generated word2vec embeddings.

#### Results
- **HMM**:
  - Produced transition matrix (`A`, 12x12) and emission matrix (`B`, 12 tags x 7 words), saved in `zamojci1_hmm.ipynb`.
  - Captured tag transition patterns (e.g., Noun→Verb) and word emissions (e.g., `science` with Noun).
- **RNN (PyTorch)**:
  - Best accuracy: 91.33% (RMSprop, varied configurations).
  - Noted variance across models (e.g., 91.33% vs. 1.62% for poor learning rates), suggesting sensitivity to hyperparameters.
- **LSTM (Keras)**:
  - Best accuracy: 96.46% (Adam, 128D embeddings, 100 hidden nodes).
  - All models scored >91%, with faster training (3 epochs vs. 100).
- **Output**: Saved code (`zamojci1_hmm.ipynb`, `zamojci1_nn.ipynb`, `zamojci1_extra.ipynb`), matrices, and accuracy tables in report.

![q1p1](https://github.com/user-attachments/assets/36264914-955b-4a34-818e-23a568642598)

![q1p2](https://github.com/user-attachments/assets/9786da0c-5ebc-4288-ab91-d9408c8d0fb0)

#### Key Skills
- Probabilistic modeling (HMM).
- Neural network design (RNN, LSTM).
- Feature engineering (embeddings, binary features).
- Hyperparameter tuning.
- Cross-framework proficiency (PyTorch, Keras).

### Homework 3: BERT Fine-Tuning and Non-Neural Text Classification

#### Description
Developed models for text classification, focusing on BERT fine-tuning and non-neural approaches to detect machine-generated text. The project included three parts: fine-tuning a pretrained BERT variant, building a non-neural model to outperform BERT, and fine-tuning BERT for the same task.

#### Approach
- **Problem 1: Fine-Tune a Pretrained Model**:
  - Followed the HuggingFace tutorial to fine-tune `distilbert-base-uncased` using PyTorch and the Transformers `Trainer` API.
  - Trained on the tutorial’s dataset for 3 epochs, keeping default parameters except for the model choice.
  - Evaluated accuracy on the test set, logging training metrics (loss, runtime, FLOPs).
- **Problem 2: Outperform BERT (Non-Neural)**:
  - Chose binary text classification (human vs. machine-generated text) on the M4GT-Benchmark dataset (~152K samples, ~65K human, balanced LLMs).
  - Implemented SVM models in scikit-learn with 3-fold cross-validation:
    - **Iteration 1**: Linear SVM with TF-IDF features (`CountVectorizer`: max_features=3000, min_df=2, max_df=0.7, ngram_range=(1,1)).
    - **Iteration 2**: Linear SVM with word2vec embeddings (100D, window=5, min_count=1) and PCA (1288 components, 0.99 variance), using 10K samples.
    - **Iteration 3**: RBF SVM with combined TF-IDF and word2vec features (3000 + 1288 dimensions), using 25K samples.
  - Evaluated precision, recall, F1, and accuracy, comparing to state-of-the-art (SOTA) BERT and non-neural models.
- **Problem 3: Fine-Tune BERT for Task**:
  - Fine-tuned `distilbert-base-uncased` on the M4GT-Benchmark for human vs. machine-generated classification.
  - Used HuggingFace’s `AutoModelForSequenceClassification` with PyTorch, training for 3 epochs.

#### Tools
- **HuggingFace Transformers**: Fine-tuned DistilBERT with PyTorch `Trainer` API.
- **scikit-learn**: Implemented SVM models with TF-IDF and vectorized features.
- **Gensim**: Generated word2vec embeddings.
- **NLTK**: Supported text preprocessing.
- **Pandas**: Managed dataset and results.
- **NumPy & scikit-learn**: Applied PCA and cross-validation.
- **Python**: Built pipelines in Jupyter notebooks.

#### Results
- **Problem 1: Fine-Tuning DistilBERT**:
  - Accuracy: 60.50% (3 epochs, training loss=0.66, runtime=33,212s, FLOPs=3.97e16).
  - Underperformed tutorial’s BERT-base-cased, possibly due to model size or data subset.
- **Problem 2: Non-Neural SVM**:
  - **Iteration 1 (TF-IDF, Linear SVM)**: Best performer, outperformed SOTA non-neural models.
  - **Iteration 2 (Word2vec+PCA, Linear SVM)**: Lower performance, likely due to 10K sample limit.
  - **Iteration 3 (TF-IDF+Word2vec, RBF SVM)**: Improved over Iteration 2 but below Iteration 1, using 25K samples.
  - TF-IDF captured key patterns; word2vec struggled with small subsets.
- **Problem 3: Fine-Tuned DistilBERT**:
  - Accuracy: 94.10%, significantly better than non-neural SVMs, aligning with SOTA BERT models.
- **Output**: Saved code (`HW3_Q1.py`, `HW3_Q2.ipynb`, `HW3_Q3.ipynb`), training logs, and result tables in report.

![Q3_model_summary](https://github.com/user-attachments/assets/80233bb0-ef02-4ad3-983b-638ab3542548)

#### Key Skills
- Transformer fine-tuning (DistilBERT).
- Non-neural model design (SVM).
- Feature engineering (TF-IDF, word2vec, PCA).
- Dataset selection and benchmarking.
- Comparative NLP evaluation.

## Skills Demonstrated
- **Natural Language Processing**:
  - Developed models for sentiment analysis (Naïve Bayes, SVM), POS tagging (HMM, RNN, LSTM), and text classification (SVM, DistilBERT), achieving high accuracies (up to 82.7% for sentiment, 96.5% for POS, 94.1% for classification).
  - Tackled diverse tasks, from probabilistic modeling to transformer fine-tuning.
- **Model Development**:
  - Built custom Naïve Bayes and HMMs with MLE, handling log-space probabilities and smoothing.
  - Designed neural models (RNN, LSTM) with word2vec and binary features.
  - Fine-tuned DistilBERT for sequence classification, adapting to new tasks.
  - Implemented non-neural SVMs with TF-IDF and word2vec, competing with SOTA.
- **Libraries and Tools**:
  - **NLTK**: Processed corpora (Movie Reviews, Brown, Treebank) and extracted features (unigrams, bigrams, frequencies).
  - **scikit-learn**: Leveraged `CountVectorizer`, `TfidfTransformer`, `MultinomialNB`, `LinearSVC`, and `SVC` for classification.
  - **PyTorch**: Built RNNs and used HuggingFace `Trainer` for DistilBERT fine-tuning.
  - **Keras**: Implemented LSTM models with efficient embeddings.
  - **HuggingFace Transformers**: Fine-tuned DistilBERT for NLP tasks.
  - **Gensim**: Generated word2vec embeddings for neural and non-neural models.
  - **Pandas & NumPy**: Managed data, matrices, and PCA transformations.
- **Feature Engineering**:
  - Engineered features like BoW, TF-IDF, word2vec, binary flags, frequency indices, and PCA-reduced embeddings.
  - Optimized parameters (`max_df`, `ngram_range`, embedding sizes) for robust performance.
- **Technical Proficiency**:
  - Combined theoretical foundations (e.g., MLE, cross-entropy, transformer architecture) with practical implementation across frameworks.
  - Conducted cross-validation, hyperparameter tuning, and comparative experiments, delivering well-documented code and reports.
