import csv

from time import time

import numpy as np
import pandas as pd
from skimage.transform import resize

from constants import ALPHABET
from constants import BASE_TARGET_FEATURES_DIRECTORY
from constants import FEATURE_EXTRACTION_OPTIONS

CHARS_LABELS_MAP = {k: v for k, v in enumerate(ALPHABET)}

CSV_FILE_PATH = 'data.csv'

# number of instances of one class for feature extraction
CLASS_NO = 1000


def get_imgarray_from_csv_file_row(row_arg):
    img_array = np.asarray(row_arg)
    img_array = img_array.reshape(28, 28)
    img_array = img_array.astype('uint8')
    img_array = resize(img_array, (20, 20))
    return img_array


# for every descriptor create file with features from every image
for feature_descriptor_name in FEATURE_EXTRACTION_OPTIONS:

    if not FEATURE_EXTRACTION_OPTIONS[feature_descriptor_name]['enabled']:
        continue

    feature_descriptor_ref = FEATURE_EXTRACTION_OPTIONS[feature_descriptor_name]['descriptor']

    feature_vectors = []

    start = time()
    with open(CSV_FILE_PATH, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        label = ALPHABET[0]
        counter = 1

        for row in reader:

            new_label = CHARS_LABELS_MAP[int(row.pop(0))]

            if new_label is label:
                if counter > CLASS_NO:
                    continue
                else:
                    img = get_imgarray_from_csv_file_row(row)

                    feature_descriptor = feature_descriptor_ref()
                    feature_descriptor.preprocess(img)
                    feature_descriptor.describe()
                    feature_vector = feature_descriptor.get_feature_vector()
                    feature_vector.append(label)

                    feature_vectors.append(feature_vector)

                    counter += 1
            else:
                label = new_label
                counter = 1

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
