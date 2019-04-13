from time import time

import csv
import string

import pandas as pd
import numpy as np

from skimage.transform import resize

from constants import BASE_TARGET_FEATURES_DIRECTORY
from constants import FEATURE_EXTRACTION_OPTIONS

ALPHABET = list(string.ascii_uppercase)

CHARS_LABELS_MAP = {k: v for k, v in enumerate(ALPHABET)}

CSV_FILE_PATH = 'data.csv'


def get_imgarray_from_csv_file_row(row_arg):
    img_array = np.asarray(row_arg)
    img_array = img_array.reshape(28, 28)
    img_array = img_array.astype('uint8')
    img_array = resize(img_array, (20, 20))
    return img_array


# for every descriptor create file with features from every image
for feature_descriptor_name in FEATURE_EXTRACTION_OPTIONS:
    feature_descriptor_ref = FEATURE_EXTRACTION_OPTIONS[feature_descriptor_name]['descriptor']

    feature_vectors = []

    start = time()
    with open(CSV_FILE_PATH, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in reader:
            label = CHARS_LABELS_MAP[int(row.pop(0))]
            img = get_imgarray_from_csv_file_row(row)

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
        print("Creating features using " + feature_descriptor_name + "occupied: ", start - end)



