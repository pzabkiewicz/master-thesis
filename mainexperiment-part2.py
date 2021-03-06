import pickle

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from constants import BASE_TARGET_FEATURES_DIRECTORY
from constants import CLASSIFIERS
from constants import FEATURE_EXTRACTION_OPTIONS

""" This script performs 10 cross validation for each combination
of feature extraction method and classifier and saves the results
in form of accuracy in a pickle. The output file will be then used 
in jupyter-notebook to perform statistical analysis with null 
hypothesis that there's no significant difference in classification
accuracy between different extraction method within a specific classifier. """

picklepath = 'mainexperiment-results/results.pickle'

try:
    with open(picklepath, 'rb') as f:
        results = pickle.load(f)
except EOFError:
    results = None

if not results:
    results = {
        'svm': {
            'zoning': [],
            'edge_maps': [],
            'local_binary_pattern': [],
            'zoning_chain_code': []
        },
        'knn': {
            'zoning': [],
            'edge_maps': [],
            'local_binary_pattern:': [],
            'zoning_chain_code': []
        },
        'mlp': {
            'zoning': [],
            'edge_maps': [],
            'local_binary_pattern:': [],
            'zoning_chain_code': []
        }
    }

for method, options in FEATURE_EXTRACTION_OPTIONS.items():

    if not options['enabled']:
        continue

    print('Fitting on data extracted with', method)

    fid = options['target_features_filename']
    filepath = BASE_TARGET_FEATURES_DIRECTORY + fid

    df = pd.read_csv(filepath)

    X = df.values[:, 1:-1]
    y = df.values[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    kfold = StratifiedKFold(n_splits=10, random_state=42).split(X_train, y_train)

    for k, (train, test) in enumerate(kfold):
        for clf, clf_options in CLASSIFIERS.items():

            classifier = clf_options['clf']

            if not clf_options['enabled']:
                continue

            classifier.fit(X_train[train], y_train[train])
            score = classifier.score(X_train[test], y_train[test])
            results[clf][method].append(score)

            print('FOLD #%d: Classification accuracy using %s: %.2f' % (k, clf, score * 100))

    with open(picklepath, 'wb') as f:
        pickle.dump(results, f)
