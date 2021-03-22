# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data used in this project is related to direct marketing campaigns of a european Bank. Using a classification model, we want to predict if a client will subscribe to a term.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
By comparing the results of a HyperDrive and an AutoML Pipeline run, we identified the best performing model for our dataset to be a VotingEnsemble with an accuracy of .9164

## Scikit-learn Pipeline
We use the Logistic Regression algorithm from Sci-KitLearn in conjunction with HyperDrive for hyperparameter tuning. The pipeline consists of the following steps:

- Data Collection (using TabularDatasetFactory and a provided link)
- Data Cleansing (code provided in train.py)
- Test - Train - Split (split 20/80, with 20% for training and 80% for testing)
- Hyperparameter Sampling (using random sampling to save time and resources)
- Model Training
- Model Testing (using the BanditPolicy as early stopping policy for computational efficiency; model performance benchmark: Accuracy)
- Saving the best-performing Model for later use
- 
We use a provided script named train.py, to control all steps except for the Hyperparameter sampling, which is controlled by HyperDrive. 

## AutoML
AutoML provides a wide variety of algorithms to work with, supporting classification, regression and time-series forecasting problems. The exit criteria is specified in order to stop the training which ensures the resources are not used once the objectives are met. This helps save on costs. Due to the fact that we were utilize a Udacity Virtual Machine for Azure we could only specify a length of 30 minutes for an experiment prior to it timing out. However we were able to iterate through the following model pipelines (with their results):

## Pipeline comparison
The difference in accuracy was miniscule, with an accuracy of 0.9086 for the Hyperdrive and 0.9164 for the AutoML run 


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Our dataset is biased toward a class. To obtain better results, we could either look for more additional data to counter the inherent bias, or do some feature engineering. 
## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
