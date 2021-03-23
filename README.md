# Optimizing an ML Pipeline in Azure

## Overview

In this project, we will build and optimize an AzureML pipeline using the Python SDK and a provided Scikit-learn model on a dataset from the internet. 
By implementing and running a Hyperdrive Pipeline and an AutoML run on the same data, we seek to obtain the best performing model in terms of accuracy. 

## Summary
The data used in this project is related to direct marketing campaigns of a banking institution, consisting of 20 feature-columns with personal data for 32950 observations and a target column "y" with the outcome of the marketing campaign. Using a classification model, we want to predict if a client will subscribe to a term, based on the feature columns.

By running and comparing the results of a HyperDrive and an AutoML Pipeline run, we identified the best performing model for our dataset to be a VotingEnsemble from the AutoML run, with an accuracy of 0.9164. 


## Scikit-learn Pipeline
We use the Logistic Regression algorithm from Sci-KitLearn in conjunction with HyperDrive for hyperparameter tuning. The pipeline consists of the following steps:

- Data Collection (using TabularDatasetFactory and a provided link)
- Data Cleansing (code provided in train.py)
- Test - Train - Split (split 20/80, with 20% for training and 80% for testing)
- Hyperparameter Sampling (using random sampling to save time and resources)
- Model Training
- Model Testing (using the BanditPolicy as early stopping policy for computational efficiency; model performance benchmark: Accuracy)
- Saving the best-performing Model for later use

We use a provided script named train.py, to control all steps except for the Hyperparameter sampling, which is controlled by HyperDrive. 

## AutoML
AutoML provides a wide variety of algorithms to work with, e.g. classification, regression and time-series forecasting problems. The exit criteria is specified in order to stop the training which to save on computational resources therefore saving cost. Due to the fact that we utilized a VM from Udacity, we could only specify a length of 30 minutes for an experiment prior to it timing out. 

## Pipeline comparison
The difference in accuracy was miniscule, with an accuracy of 0.9086 for the HyperDrive and 0.9164 for the AutoML run. The HyperDrive was faster than the AutoML run though.   


## Future work

There are several measures that could be taken to get better results. 
We might make improvements to the HyperDrive and the AutoML run. One could use the recursive Bayesian Parameter Sampling instead of Random Sampling with the HyperDrive. Bayesian Parameter Sampling ould require more computational resources since we can´t specify an early termination policy, but might get superior results in terms of accuracy. The AutoML could be improved by adjusting the experiment timeout to more than 30 minutes which would allow for more model experimentation. 
Also, we could work on the dataset, which is biased toward a class. To obtain better results, we could either look for additional data to counter the inherent bias, we could do some feature engineering, or we could try techniques like oversampling of the minority class to get a balanced dataset. 


