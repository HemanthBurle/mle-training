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


## To excute the script
``python < scriptname.py >``


## Command to create an environment from the env.yml file you have generated. This ensures reproducibility of the exact same environment that you have created.
``conda env create -f env.yml``

## Command to activate the environment
``conda acrivate mle-dev``

## Command to run the python script.
``python nonstandardcode.py``
