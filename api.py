import numpy as np
import pandas as pd

# import local scripts
from training import train

if __name__ == '__main__':
    # user_input = input("which process: ")
    user_input = 't'
    # data = input("which dataset: ")
    data = 'kitsune SSL'
    model_type = input("which model: ") 
    label_col = ['label']
    if user_input == "t":
        print("train process selected")
        train(data, model_type, False, label_col)
    elif user_input == "p":
        print("predict process selected")
        train(data, model_type, True, label_col)
    else:
        raise Exception("process not defined")