import pandas as pd

from time import time

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from constants import BASE_TARGET_FEATURES_DIRECTORY
from constants import FEATURE_EXTRACTION_OPTIONS

""" This script performs grid search to find parameters of best classifier. 
These parameters will be then used in main experiments.  """


PARAM_RANGE_SVM_KNN = [0.0001 * 10**i for i in range(9)]

CLASSIFIER_OPTIONS = {
    'svm': {
        'estimator': SVC(cache_size=7000, max_iter=1000, random_state=42),
        'parameters': [
            {
                'kernel': ['linear'],
                'C': PARAM_RANGE_SVM_KNN,
            },
            {
                'kernel': ['rbf'],
                'C': PARAM_RANGE_SVM_KNN,
                'gamma': PARAM_RANGE_SVM_KNN
            }
        ],
        'enabled': True
    },
    'knn': {
        'estimator': KNeighborsClassifier(),
        'parameters':
            {
                'n_neighbors': [3, 4, 5],
                'metric': ['euclidean', 'manhattan', 'chebyshev'],
            },
        'enabled': True
    },
    'mlp': {
        'estimator': MLPClassifier(random_state=42),
        'parameters':
            {
                 'hidden_layer_sizes': [(i*50,) for i in range(1, 11)],
                 'activation': ['logistic', 'tanh', 'relu']
            },
        'enabled': True
    }
}

ZONING_CHAIN_CODE_FEATURES_FILEPATH = BASE_TARGET_FEATURES_DIRECTORY + \
                                      FEATURE_EXTRACTION_OPTIONS['zoning_chain_code']['target_features_filename']

df = pd.read_csv(ZONING_CHAIN_CODE_FEATURES_FILEPATH)

X = df.values[:, 1:-1]
y = df.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for classifier_name in CLASSIFIER_OPTIONS:

    if not CLASSIFIER_OPTIONS[classifier_name]['enabled']:
        continue

    classifier = CLASSIFIER_OPTIONS[classifier_name]['estimator']

    start = time()

    gs = GridSearchCV(estimator=classifier,
                      param_grid=CLASSIFIER_OPTIONS[classifier_name]['parameters'],
                      scoring='accuracy',
                      cv=10,
                      n_jobs=3,
                      verbose=10)

    gs = gs.fit(X_train, y_train)

    end = time()

    print(5 * '#' + classifier_name.upper() + 5 * '#', end='\n')
    print('Mean accuracy of best classifier: ', gs.best_score_)
    print('Parameters of best classifier: ', gs.best_params_, end='\n')
    print('GridSearch occupied: ', end - start, end='\n')
