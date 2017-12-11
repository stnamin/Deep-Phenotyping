# Deep-Phenotyping
Python codes for deep phenotyping using Theano and Keras


## Our paper
Please reference our original paper when using these codes.

Sarah Taghavi Namin, Mohammad Esmaeilzadeh, Mohammad Najafi, Tim B. Brown, and Justin O. Borevitz, 'Deep Phenotyping: Deep Learning for Temporal Phenotype/Genotype Classification', bioRxiv, 2017.

## Files description

cnn.py: Train and test with cnn (alexnet)

cnn_lstm.py: extract deep features using cnn, train and test lstm (label last frame)

cnn_lstm_perframe.py: extract deep features using cnn, train and test lstm (label each frame of sequence)

cnn_lstm_perframe_train.py: extract deep features using cnn, train lstm with all the data

cnn_lstm_perframe_test_rest_accessions.py: classify other accessions 

cnn_featuremaps.py: feature maps form differnt layers of cnn

cnn_crf.py: extract deep features using cnn, train and test with CRF instead of lstm for temporal info

hcf_svm.py: handcrafted features, using svm for classification

hcf_lstm.py: handcrafted features, train and test with lstm

hcf_important_features.py: finding important handcrafted features

pots.py: preparing pots data

pots_with_area.py: preparing pots data + area 

pots_with_more_features.py: preparing pots data + handcrafted features

pots_with_more_features_with_fourier.py: preparing pots data + handcrafted features + fourier features

Grabcut_segmentation_more_features_and_fourier.py: Segmentation and handcrafted features extraction 

## Dataset

https://figshare.com/s/e18a978267675059578f or http://phenocam.anu.edu.au/cloud/a_data/_webroot/published-data/2017/2017-Namin-et-al-DeepPheno.zip
