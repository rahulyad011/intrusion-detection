import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing library for support vector machine classifier
from sklearn.svm import SVC

# MLP requirements 
from keras.layers import Dense # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.models import model_from_json # saving and loading trained model

# LSTM requirements 
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model

# representation of model layers
from keras.utils.vis_utils import plot_model


# import local scripts
from training import train

if __name__ == '__main__':
    user_input = input("which process: ")
    # model_type = input("which model: ")
    data = input("which dataset: ")
    if user_input == "train":
        print("train process selected")
        train(data)
    elif user_input == "prediction":
        print("prediction process selected")
    else:
        raise Exception("process not defined")