import pandas as pd

from os import listdir
from time import time

from skimage.io import imread
from skimage.transform import resize

from constants import BASE_IMAGES_DIRECTORY
from constants import BASE_TARGET_FEATURES_DIRECTORY
from constants import FEATURE_EXTRACTION_OPTIONS

images_subdirectories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
filepaths = []

# read all filenames
for image_subdirectory in images_subdirectories:
    for file_id in listdir(BASE_IMAGES_DIRECTORY + image_subdirectory):

        filepath = BASE_IMAGES_DIRECTORY + image_subdirectory + '/' + file_id
        filepaths.append(filepath)

# for every descriptor create file with features from every image
for feature_descriptor_name in FEATURE_EXTRACTION_OPTIONS:
    feature_descriptor_ref = FEATURE_EXTRACTION_OPTIONS[feature_descriptor_name]['descriptor']

    feature_vectors = []

    start = time()
    for filepath in filepaths:
        label = filepath.split('/')[1]  # name of char
        img = imread(filepath)
        img = resize(img, (20, 20))

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
    target_features_filename = BASE_TARGET_FEATURES_DIRECTORY +\
                               FEATURE_EXTRACTION_OPTIONS[feature_descriptor_name]['target_features_filename']
    df.to_csv(target_features_filename)

    end = time()
    print("Creating features using " + feature_descriptor_name + ": ", start-end)











