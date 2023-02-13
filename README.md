# Credit Risk Analysis

## Overview

The purpose of this analysis is to use several machine learning algorithms to predict patterns based on the data provided. In this assignment, we were exploring the riskiniess of a credit loan. Credit risk is an inherently unbalanced classification problem because good loans easily outnumber risky loans.

In the dataset provided to us by LendingHome, a peer-to-peer lending services company, we utilized supervised learning. Supervised learning includes data with a labeled outcome; the credit risk outcome is outlined in the dataset. Then, we employed machine learning to oversample the data, undersample the data, and explore models that reduce bias. 

## Results

We used six machine learning algorithms, all easily available to us in Python with the scikit-learn and imbalanced-learn libraries.

### RandomOverSampler Model (Oversampling)

![Screen Shot 2023-02-13 at 7 38 34 AM](https://user-images.githubusercontent.com/112633146/218459773-9491a80b-357a-4c51-85fd-46905e685f43.png)

- Balance accuracy: 0.6443721269403855
- Precision: The precision rate for high-risk loans is only 1%, and for low-risk loans, it is 100%.
- Recall score: high/low: .62/.68


### SMOTE (Synthetic Minority Oversampling Technique) Model

![Screen Shot 2023-02-13 at 7 39 53 AM](https://user-images.githubusercontent.com/112633146/218459990-d0a17da4-2c1e-4cef-ad58-561806fc3d47.png)

- Balance accuracy: 0.6443721269403855
- Precision: The precision rate for high-risk loans still remains 1%, and for low-risk loans, it is 100%.
- Recall score: high/low: .63/.66


### ClusterCentroids Model (Undersampling)

![Screen Shot 2023-02-13 at 7 49 36 AM](https://user-images.githubusercontent.com/112633146/218461950-afdde7ae-0a62-42eb-b716-ef4bd81ebe30.png)

- Balance accuracy: 0.6443721269403855
- Precision: The precision rate for high-risk loans is 1%, and for low-risk loans, it is 100%.
- Recall score: high/low: .61/.45

### SMOTEE-NN (Synthetic Minority Oversampling Technique - NearestNeighbors) (Combination Sampling)

![Screen Shot 2023-02-13 at 7 51 35 AM](https://user-images.githubusercontent.com/112633146/218463095-84cee12c-6834-4b0c-8bd6-ab77508cdee1.png)

- Balance accuracy: 0.5292150629907619
- Precision: The precision rate for high-risk loans is 1%, and for low-risk loans, it is 100%.
- Recall score: high/low: .70/.57

### BalancedRandomForestClassifier Model (Bias Reduction)

![Screen Shot 2023-02-13 at 7 52 52 AM](https://user-images.githubusercontent.com/112633146/218463387-bfdcdfec-fc1b-408b-b729-4666389eee80.png)

- Balance accuracy: 0.7877672625306695
- Precision: The precision rate for high-risk loans is 4%, and for low-risk loans, it is 100%.
- Recall score: high/low: .67/.91

### EasyEnsembleClassifier Model (Bias Reduction)

![Screen Shot 2023-02-13 at 7 57 47 AM](https://user-images.githubusercontent.com/112633146/218464412-963bb520-b8e1-4055-b4f5-8690ae026a32.png)

- Balance accuracy: 0.925427358175101
- Precision: The precision rate for high-risk loans is 7%, and for low-risk loans, it is 100%.
- Recall score: high/low: .91/.94


## Summary

When observing balanced accuracy and evaluating the strength of each model, we want the balance accuracy score which is closest to 1. With this rule, the EasyEnsembleClassifer Model is the best machine learning model with .925 as the balanced accuracy. Across all of the models, there is not a significant difference in precision, as the precision for all low-risk loans is 100% and between 1% and 7% for high-risk loans. When observing the recall score, the closer the number is to 1, the better fit the machine learning model is. Once again, the EasyEnsembleClassifier Model has the highest recall score at .94. This model is the best all-around machine learning model for this data.


