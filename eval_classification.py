# Experiment scripts for binary classification benchmarks (e.g. MR, CR, MPQA, SUBJ)

from time import process_time, time
import numpy as np
import pickle
from joblib import dump, load
import importlib
from os import getcwd
from os.path import join, isfile
st = importlib.import_module("skip-thoughts")

from scipy.sparse import hstack

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

def eval_nested_kfold(encoder, name, loc='./data/', k=5, seed=1234, use_nb=False):
    """
    Evaluate features with nested K-fold cross validation
    Outer loop: Held-out evaluation
    Inner loop: Hyperparameter tuning

    Datasets can be found at http://nlp.stanford.edu/~sidaw/home/projects:nbsvm
    Options for name are 'MR', 'CR', 'SUBJ' and 'MPQA'
    """
    # Load the dataset and extract features
    z, features = st.dataset_handler.load_data(encoder, name, loc=loc, seed=seed)
    
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'sentence_embeddings_' + time_stamp + '.dat'
    file_name_full_path = join(loc, 'skip-thoughts', 'data', file_name)
    print("Saving embeddings to file {0}".format(file_name_full_path))
    with open(file_name_full_path, 'wb') as f:
        pickle.dump(z, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    scan = [2**t for t in range(0,3,1)]
    #npts = len(z['text'])
    #kf = KFold(npts, n_folds=k, random_state=seed)
    kf = KFold(n_splits=k, random_state=seed)
    
    start_time = process_time()
    print("Started 'eval_nested_kfold'".format(start_time))
    
    scores = []
    for train_index, test_index in kf.split(features):

        # Split data
        X_train, y_train = features[train_index], z['labels'][train_index]
        X_test, y_test = features[test_index], z['labels'][test_index]


        Xraw = [z['text'][i] for i in train_index]
        Xraw_test = [z['text'][i] for i in test_index]

        scanscores = []
        for s in scan:

            # Inner KFold
            innerkf = KFold(n_splits=k, random_state=seed+1)
            innerscores = []
            for innertrain, innertest in innerkf.split(X_train):
        
                # Split data
                X_innertrain = X_train[innertrain]
                y_innertrain = y_train[innertrain]
                X_innertest = X_train[innertest]
                y_innertest = y_train[innertest]

                Xraw_innertrain = [Xraw[i] for i in innertrain]
                Xraw_innertest = [Xraw[i] for i in innertest]

                # NB (if applicable)
                if use_nb:
                    NBtrain, NBtest = compute_nb(Xraw_innertrain, y_innertrain, Xraw_innertest)
                    X_innertrain = hstack((X_innertrain, NBtrain))
                    X_innertest = hstack((X_innertest, NBtest))

                # Train classifier
                clf = LogisticRegression(C=s)
                clf.fit(X_innertrain, y_innertrain)
                acc = clf.score(X_innertest, y_innertest)
                innerscores.append(acc)
                print (s, acc)

            # Append mean score
            scanscores.append(np.mean(innerscores))

        # Get the index of the best score
        s_ind = np.argmax(scanscores)
        s = scan[s_ind]
        print (scanscores)
        print (s)
 
        # NB (if applicable)
        if use_nb:
            NBtrain, NBtest = compute_nb(Xraw, y_train, Xraw_test)
            X_train = hstack((X_train, NBtrain))
            X_test = hstack((X_test, NBtest))
       
        # Train classifier
        clf = LogisticRegression(C=s)
        clf.fit(X_train, y_train)
        
        cwd = getcwd()
        dump(clf, join(cwd,'skip-thoughts/models', 'best_logit.joblib.gz'))
        
        # Evaluate
        acc = clf.score(X_test, y_test)
        scores.append(acc)
        print (scores)

        end_time = process_time()
        
        print("Elapsed time in seconds for 5-fold CV and hyperparameter tuning on Logit: {0:4.2f}".format(end_time - start_time))    

    return scores


def compute_nb(X, y, Z):
    """
    Compute NB features
    """
    labels = [int(t) for t in y]
    ptrain = [X[i] for i in range(len(labels)) if labels[i] == 0]
    ntrain = [X[i] for i in range(len(labels)) if labels[i] == 1]
    poscounts = st.nbsvm.build_dict(ptrain, [1,2])
    negcounts = st.nbsvm.build_dict(ntrain, [1,2])
    dic, r = st.nbsvm.compute_ratio(poscounts, negcounts)
    trainX = st.nbsvm.process_text(X, dic, r, [1,2])
    devX = st.nbsvm.process_text(Z, dic, r, [1,2])
    return trainX, devX


def eval_test_data(encoder, name, loc='./data/'):
    """
    load previously saved logistic regression model to predict on test data.
    Only works on ACLIMBD, because it has train and test data stored separately.
    """
    acc = 0.0
    if name == 'ACLIMBD':
        full_path_model_file = join(loc, 'skip-thoughts', 'models', 'best_logit.joblib.gz')
        if isfile(full_path_model_file):
            full_path_test = join(loc, 'skip-thoughts', 'data', 'aclImdb', 'test')
            z, features = st.dataset_handler.load_data(encoder, name, loc=full_path_test)
            clf = load(full_path_model_file)
            acc = clf.score(features, z['labels'])
    return acc