# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## How to run the code
 - Open miniconda prompt
 - Change directory to project folder
 - Type the below line to replicate the environment created
   - conda env create -f env.yml
 - Now change directory to src folder
 - Type the below line to execute the code
   - python ingest_data.py --> to create the dataset for further process, else default dataset will be used
   - python train.py --> to train the datasets available, else default trained model will be used
   - python score.py --> to display the scores of different models used
 - Alternatively
   - if you want to change output directory for any code
     - python "code name".py -p "directory name"
   - if you want to change log path for any code
     - python "code name".py --logpath "log path directory"

## P.S. Whole process takes about 5-10 min
