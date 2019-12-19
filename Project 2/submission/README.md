# EPFL Machine Learning Project 2 README

##Packages:  

Our code use the following packages you will need to install these packages before running it, instructions for installation ara available on their respectives websites. We use the following version :

- numpy: Version: 1.16.2

- pandas: Version: 0.24.2

- scikit-learn: Version: 0.20.3

- scikit-surprise: Version: 1.1.0

For the surprise package you can install it using the following command:
- pip install scikit-surprise
- conda install -c conda-forge scikit-surprise

## Files

This submission€™ folder contains the following 9 elements:

- project_helpers.py: Contains all functions needed in the project, such as the function to convert the dataset in a format we can use with surprise package.

- Gread_search.ipynb: The jupyter notebook we use to run our gread search for each model, this script was run on a Google Cloud machine take a lot of time to run.

- Blending.ipynb: The jupyter notebook we use to blend each model together with the best hyperparameters we find using the Gread_search.ipynb script. 

- run.ipynb: The script that generate the file Submission.csv in a jupyter notebook we use the best hyperparameters that we determine using Gread_search.ipynb and Blending.ipynb scripts. 

- run.py: The script that generate the file Submission.csv 

- Figures Folder: Contains the 3 images used in the report

- data_train.csv: train dataset provide for our project 

- sample.Submission.csv: test dataset provide for our project

- Submission.csv: This .csv file contains the predictions obtained by running the run.py script. This is the prediction file that has been submitted to AICrowd and gave the final best score.

##Computer configuration 

All our codes take a lot of time to run especially our Gread_search.ipynd file to improve our computation power and reduce the computation time we use a virtual machine provide by Google Cloud Platfom its configuration is detailed bellow: 
n1-standard-16 :16 vCPU, 60 Go  ubuntu-1604-xenial-v20191113