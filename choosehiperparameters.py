import pandas as pd

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from constants import BASE_TARGET_FEATURES_DIRECTORY
from constants import FEATURE_EXTRACTION_OPTIONS

PARAM_RANGE = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CLASSIFIER_OPTIONS = {
    'svm': {
        'estimator': SVC(random_state=42),
        'parameters': [
            {
                'kernel': ['linear'],
                'C': PARAM_RANGE,
            },
            {
                'kernel': ['rbf'],
                'C': PARAM_RANGE,
                'gamma': PARAM_RANGE
            }
        ]
    },
    'knn': {
        'estimator': KNeighborsClassifier(random_state=42),
        'parameters':
            {
                'n_neighbors': [3, 4, 5],
                'metric': ['euclidean', 'manhattan', 'chebyshev'],
            }
    },
    'mlp': {
        'estimator': MLPClassifier(random_state=42),
        'parameters': None
    }
}

ZONING_CHAIN_CODE_FEATURES_FILEPATH = BASE_TARGET_FEATURES_DIRECTORY + \
                                      FEATURE_EXTRACTION_OPTIONS['zoning_chain_code']['target_features_filename']

df = pd.read_csv(ZONING_CHAIN_CODE_FEATURES_FILEPATH)

X = df.values[:, :-1]
y = df.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for classifier_name in CLASSIFIER_OPTIONS:
    classifier = CLASSIFIER_OPTIONS[classifier_name]['estimator']

    gs = GridSearchCV(estimator=classifier,
                      param_grid=CLASSIFIER_OPTIONS[classifier_name]['parameters'],
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)

    gs = gs.fit(X_train, y_train)

    print(5 * '#' + classifier_name.upper() + 5 * '#', end='\n')
    print('Sredni wynik z 10-krotnej walidacji krzyzowej: ', gs.best_score_)
    print('Parametry najelepszego klasyfikatora: ', gs.best_params_, end='\n')
