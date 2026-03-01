import pandas as pd 
import numpy as np 
from config import train_path , test_path
def load_(train_path,test_path):
    training_data = pd.read_csv(train_path)
    testing_data = pd.read_csv(test_path)

    X_train , y_train = training_data.iloc[:,1:].values , training_data.iloc[:,0].values
    X_test , y_test = testing_data.iloc[:,1:].values , testing_data.iloc[:,0].values
    print("Code Works")

    return X_train , X_test , y_train , y_test


if __name__ == "__main__":
    load_(train_path,test_path)
