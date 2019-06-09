This code was written for making study on following subject: _Feature extraction for computer aided recognition of structurized handwritting_.

Within the scope of this master thesis a feature extraction method was proposed. This method makes use of two other algorithms: _Freeman chain code_ and one of modification of _zoning_. This method was compared with 3 others algorithms and namely: _Local Binary Pattern_, _Zoning_, _Edge Maps_ (references are cited in code).

Following steps were conducted within the master thesis:

1. Feature extraction with 4 feature extraction methods from _.csv_ file with image matrices downloaded from kaggle.
2. Grid search on 3 classifiers (SVM, KNN, MLP) on data extracted with own feature extraction method.
3. Generating of confusion matrices for each combination: feature extraction method - classifier.
4. Statistical analysis with null hypothesis: _There is no difference in classification accuracy of 26 capital english chars between 4 feature extraction methods witihin one classifier_.
