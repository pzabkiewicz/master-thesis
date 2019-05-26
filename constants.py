import string

from featuredescriptors.featuredescriptors import ZoningChainCode
from featuredescriptors.featuredescriptors import LocalBinaryPattern
from featuredescriptors.featuredescriptors import EdgeMaps
from featuredescriptors.featuredescriptors import Zoning

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ALPHABET = list(string.ascii_uppercase)

BASE_IMAGES_DIRECTORY = 'images/'
BASE_TARGET_FEATURES_DIRECTORY = 'features_csv/'

FEATURE_EXTRACTION_OPTIONS = {
    'zoning': {
        'descriptor': Zoning,
        'target_features_filename': 'zoning_features.csv',
        'enabled': False
    },
        'edge_maps': {
        'descriptor': EdgeMaps,
        'target_features_filename': 'edge_maps_features.csv',
        'enabled': False
    },
        'local_binary_pattern': {
        'descriptor': LocalBinaryPattern,
        'target_features_filename': 'local_binary_pattern.csv',
        'enabled': False
    },
        'zoning_chain_code': {
        'descriptor': ZoningChainCode,
        'target_features_filename': 'zoning_chain_code.csv',
        'enabled': True
    }
}

# Classifiers with hiperparameters chosen based on grid search
# They are used for parts of the main experiment
CLASSIFIERS = {
    'svm': {
        'clf': SVC(kernel='rbf', C=10.0, gamma=0.1, random_state=42),
        'enabled': True
    },
    'knn': {
        'clf': KNeighborsClassifier(metric='manhattan', n_neighbors=5),
        'enabled': True,
    },
    'mlp': {
        'clf': MLPClassifier(hidden_layer_sizes=(300,), activation='tanh', random_state=42),
        'enabled': True
    }
}