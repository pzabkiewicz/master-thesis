import string
from os import listdir
from random import random
from time import time

import pandas as pd
from skimage.io import imread
from skimage.transform import resize

from constants import BASE_IMAGES_DIRECTORY
from constants import BASE_TARGET_FEATURES_DIRECTORY
from constants import FEATURE_EXTRACTION_OPTIONS

ALPHABET = list(string.ascii_uppercase)

CHARS_LABELS_MAP = {k: v for k, v in enumerate(ALPHABET)}

CSV_FILE_PATH = 'data.csv'

CHAR_DIRECTORIES = listdir(BASE_IMAGES_DIRECTORY)

# number of instances of one class for feature extraction
CLASS_NO = 500

chosen_char_file_paths = []

# choose random chars from char directory for dataset
for char_directory in CHAR_DIRECTORIES:

    img_filenames = listdir(BASE_IMAGES_DIRECTORY + char_directory)

    counter = 0

    while counter < CLASS_NO and len(img_filenames) > 0:

        idx = int(random() * len(img_filenames))
        fid = img_filenames.pop(idx)

        chosen_char_file_paths.append(char_directory + '/' + fid)

        counter += 1


# for every descriptor create file with features from every image
for feature_descriptor_name in FEATURE_EXTRACTION_OPTIONS:

    if not FEATURE_EXTRACTION_OPTIONS[feature_descriptor_name]['enabled']:
        continue

    feature_descriptor_ref = FEATURE_EXTRACTION_OPTIONS[feature_descriptor_name]['descriptor']

    feature_vectors = []

    start = time()

    for char_file_path in chosen_char_file_paths:

        label = char_file_path[0]

        img = imread(BASE_IMAGES_DIRECTORY + char_file_path)

        feature_descriptor = feature_descriptor_ref()
        feature_descriptor.preprocess(img)
        feature_descriptor.describe()
        feature_vector = feature_descriptor.get_feature_vector()
        feature_vector.append(label)

        feature_vectors.append(feature_vector)


    # creating data frame and saving as .csv
    column_names = ['f#' + str(col_name) for col_name in range(len(feature_vector[:-1]))]
    column_names.append('class')

    df = pd.DataFrame(data=feature_vectors, columns=column_names)
    target_features_filename = BASE_TARGET_FEATURES_DIRECTORY + \
                               FEATURE_EXTRACTION_OPTIONS[feature_descriptor_name]['target_features_filename']
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    df.to_csv(target_features_filename)

    end = time()
    print("Creating features using " + feature_descriptor_name + " occupied: ", end - start)
