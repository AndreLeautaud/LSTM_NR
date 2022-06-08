# LSTM_June2022
Long-short term memory neural network for n2o emissions 

Folders are numbered in order of use. 

Datasets: 
* GlobalN2ODB_NN_cleaned_dataset - 69 site-treatments currently (6/8/22) being used for training/testing of the model 
* GlobalN2ODB_NN_Holdout_datasets - 4 site-treatments being used as holdout (never seen by model) data sets 

1 - Matlab processing: 
This preps the datasets for use in python
Matlab scripts should have n2osf and SL as variables in it, and otherwise work per model (need to generate script for other models still)    
* base_June2022 - the 4 covariate model 
* covariate7_dp_June2022 - 7 covariate version of the model 

2 - matlab data
XTrain/XVal/YTrain/Yval/mu/sigma get saved here per model 

3 - python model training 
Build the LSTM here per model 

4 - python models 
HDF5 LSTM models get saved here 

5 - estimation 
Python script that uses said models to run across train/test and holdout sites, generating results 

6 - outputs, error tables  
Error tables from model results 

7 - outputs, timeseries data
daily measured-modeled results from model  

8 - outputs, plots  
png results for daily time series 

R_analysis 
Analyzing and comparing results from all models in R 


