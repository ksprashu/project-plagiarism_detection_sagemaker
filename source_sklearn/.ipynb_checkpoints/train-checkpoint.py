from __future__ import print_function

import argparse
import os
import pandas as pd
import numpy as np

import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--n_estimators', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    ## --- Your code here --- ##
    
    # shuffle the dataset
    train_x = train_x.sample(frac=1)    
    
    # scale the data and split into training and validation data
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x.to_numpy())
    
    # split into training and validation set for cross validation
    train_ratio = 0.8
    train_size = int(train_x.shape[0] * train_ratio)

    # get the validation data
    val_x = train_x[train_size:, :]
    val_y = train_y[train_size:]

    # get the training data
    train_x = train_x[:train_size, :]
    train_y = train_y[:train_size]

    ## TODO: Define a model 
    model = AdaBoostClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate)
    
    ## TODO: Train the model
    model.fit(train_x, train_y)
        
    # print the accuracy for hyperparam tuning
    print('accuracy = {};'.format(model.score(val_x, val_y)))        
    
    ## --- End of your code  --- ##
    
    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
