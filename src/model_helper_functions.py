import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
from pandas import DataFrame, concat
from preprocess_helper_functions import *
from sklearn.model_selection import StratifiedShuffleSplit
import os

def pr(y_i, y, x):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_model(x, y):
    r = np.log(pr(1,y, x) / pr(0,y,x))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

def call_NB_SVM_algorithm(X_train, y_train, X_test, y_test):
    m,r = get_model(X_train, y_train)
    p_test = m.predict_proba(X_test.multiply(r))[:,1]
    npround = np.vectorize(round)
    p_test_ints = npround(p_test)
    f1score = f1_score(y_test, p_test_ints)
    logloss = log_loss(list(y_test), list(p_test_ints.astype(int)))
    accuracy = accuracy_score(y_test, p_test_ints)
    return DataFrame({'pred': p_test, 'truth': y_test})
    
#----------------------------------------------------------------------------------------
#Logistic Regression
def call_logreg_algorithm(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    p_test = logreg.predict_proba(X_test)[:,1]
    npround = np.vectorize(round)
    p_test_ints = npround(p_test)
    f1score = f1_score(y_test, p_test_ints)
    #print("F1-score: ", f1score)
    logloss = log_loss(list(y_test), list(p_test_ints.astype(int)))
    #print("Log Loss: ", logloss)
    accuracy = accuracy_score(y_test, p_test_ints)
    #print("Accuracy: ", accuracy)
    return DataFrame({'pred': p_test, 'truth': y_test})

#----------------------------------------------------------------------------------------
#XGBoost 
def call_xgboost_algorithm(xgb, vectorizer, X_train, y_train, X_test, y_test):
    #Training on XGBoost
    d_train = xgb.DMatrix(X_train, label=y_train)
    #Set our parameters for xgboost
    params = {}
    num_round = 500
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = ['logloss']
    params['eta'] = 0.01 #Learning rate 
    params['max_depth'] = 6 #Depth of the tree. Default is 6. 
    params['colsample_bytree'] = 0.8
    
    bst = xgb.train(params, d_train, num_round, verbose_eval= True)
    d_test = xgb.DMatrix(X_test, label=y_test)
    p_test = bst.predict(d_test)
    
    npround = np.vectorize(round)
    p_test_ints = npround(p_test)
    f1score = f1_score(y_test, p_test_ints)
    #print("F1-score: ", f1score)
    logloss = log_loss(list(y_test), list(p_test_ints.astype(int)))
    #print("Log Loss: ", logloss)
    accuracy = accuracy_score(y_test, p_test_ints)
    #print("Accuracy: ", accuracy)
    return DataFrame({'pred': p_test, 'truth': y_test})


#----------------------------------------------------------------------------------------
#FastText 
from keras.layers import Dense, Embedding, Dropout, LSTM, Bidirectional, GlobalMaxPool1D, InputLayer, BatchNormalization, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import text, sequence

def get_fasttext_model(embedding_matrix):
    embed_size = 100
    model = Sequential([
        InputLayer(input_shape=(100,), dtype='int32'),
        Embedding(len(embedding_matrix), embed_size),
        Bidirectional(LSTM(50, return_sequences=True)),
        GlobalMaxPool1D(),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    embedding = model.layers[1]
    embedding.set_weights([embedding_matrix])
    embedding.trainable = False
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def call_fasttext_algorithm(X_train, y_train, X_test, y_test, embedding_matrix):
    batch_size = 64
    epochs = 3
    model = get_fasttext_model(embedding_matrix)
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,  verbose=True)
    p_test = model.predict(X_test, batch_size=batch_size).flatten()
    npround = np.vectorize(round)
    p_test_ints = npround(p_test)
    f1score = f1_score(y_test, p_test_ints)
    logloss = log_loss(list(y_test), list(p_test_ints.astype(int)))
    accuracy = accuracy_score(y_test, p_test_ints)
    return DataFrame({'pred': p_test, 'truth': y_test})


def preprocess_data_for_fasttext(texts_train, texts_test, train):
    with open('fasttext-embedding-train.txt', 'w', encoding='utf-8') as target:
        for text in texts_train:
            target.write('__label__0\t{0}\n'.format(text.strip()))

    os.system('../fastText-0.1.0/fasttext skipgram -input fasttext-embedding-train.txt -output embedding-model >nul 2>&1')

    text_tokens = sorted(set(' '.join(train.values).split()))

    with open("fasttext-words.txt", "w", encoding="utf-8") as target:
        for word in text_tokens:
            target.write("{0}\n".format(word.strip()))

    os.system('../fastText-0.1.0/fasttext print-word-vectors embedding-model.bin < fasttext-words.txt > fasttext-vectors.txt')

    embedding_matrix = np.zeros([len(text_tokens) + 1, 100])
    word2index = {}
    with open("fasttext-vectors.txt", "r", encoding="utf-8") as src:
        for i, line in enumerate(src):
            parts = line.strip().split(' ')
            word = parts[0]
            vector = map(float, parts[1:])
            word2index[word] = len(word2index)
            embedding_matrix[i] = np.fromiter(vector, dtype=np.float)

    def text2sequence(text):
        return list(map(lambda token: word2index.get(token, len(word2index) - 1), str(text).split()))
    X_train = sequence.pad_sequences(list(map(text2sequence, texts_train)), maxlen=100)
    X_test = sequence.pad_sequences(list(map(text2sequence, texts_test)), maxlen=100)
    return X_train, X_test, embedding_matrix

#--------------------------------------------------------------------------

def extract_combined_results(results):
    """
    Get the scores 
    """
    results_df = concat(results)
    results_df = results_df.reset_index(drop = True)
    pred_orig, truth = results_df.pred, results_df.truth
    pred = pred_orig.round()
    
    #Get the scores 
    f1score = f1_score(truth, pred)
    logloss = log_loss(truth, pred.astype(int))
    accuracy = accuracy_score(truth, pred)
    conf_mat = confusion_matrix(truth, pred)
    class_report = classification_report(truth, pred, digits=5)
    fpr, tpr, thresholds = roc_curve(truth, pred_orig)
    auc_score = auc(fpr, tpr)
    num_missclassified = sum(pred != truth)

    print("F1-score: ", f1score)
    print("Log Loss: ", logloss)
    print("Accuracy: ", accuracy)
    print("AUC score: ", auc_score)
    print("Num of comments missclassified: ", num_missclassified)
    print(conf_mat)
    print(class_report)
    
    return (f1score, logloss, accuracy, conf_mat, class_report, num_missclassified, (truth, pred_orig))



#-------------------------------------------------------------------------



