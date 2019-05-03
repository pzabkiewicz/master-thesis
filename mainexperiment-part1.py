import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from constants import BASE_TARGET_FEATURES_DIRECTORY
from constants import FEATURE_EXTRACTION_OPTIONS
from constants import ALPHABET
from constants import CLASSIFIERS

from helpers import plot_confusion_matrix

""" This script is used for generating confusion matrix for
all combinations of each classifier with each feature extraction method """

labels_conversion = {k: v for v, k in enumerate(ALPHABET)}

# loop through 'feature extraction methods' (rather through data sets with feature vectors)
for method, options in FEATURE_EXTRACTION_OPTIONS.items():

    if not options['enabled']:
        continue

    fid = options['target_features_filename']
    filepath = BASE_TARGET_FEATURES_DIRECTORY + fid

    df = pd.read_csv(filepath)

    X = df.values[:, 1:-1]
    y = df.values[:, -1]
    y = np.array([labels_conversion[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for clf in CLASSIFIERS:

        clf_options = CLASSIFIERS[clf]
        classifier = clf_options['clf']

        if not clf_options['enabled']:
            continue

        print('Fitting using', clf, 'on train data extracted with', method)
        classifier.fit(X_train, y_train)

        print('Prediction on test data...')
        y_pred = classifier.predict(X_test)

        conf_mat = confusion_matrix(y_test, y_pred)
        print('Confusion matrix: ', end='\n')
        print(conf_mat)
        print()

        # plot confusion matrix
        title = clf + ' / ' + method
        plot_confusion_matrix(y_test, y_pred, np.array(ALPHABET), normalize=True, title=title)

        target_path = 'mainexperiment-results/' + clf + '-' + method + '.eps'
        plt.savefig(target_path, format='eps', dpi=1000)
