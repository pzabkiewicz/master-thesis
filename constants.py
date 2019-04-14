from featuredescriptors.featuredescriptors import ZoningChainCode
from featuredescriptors.featuredescriptors import LocalBinaryPattern
from featuredescriptors.featuredescriptors import EdgeMaps
from featuredescriptors.featuredescriptors import Zoning

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
        'enabled': True
    },
        'local_binary_pattern': {
        'descriptor': LocalBinaryPattern,
        'target_features_filename': 'local_binary_pattern.csv',
        'enabled': True
    },
        'zoning_chain_code': {
        'descriptor': ZoningChainCode,
        'target_features_filename': 'zoning_chain_code.csv',
        'enabled': True
    }
}