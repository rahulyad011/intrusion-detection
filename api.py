import numpy as np
import pandas as pd

# import local scripts
from training import train

if __name__ == '__main__':
    user_input = input("which process: ")
    data = input("which dataset: ")
    model_type = input("which model: ")
    if user_input == "train":
        print("train process selected")
        train(data, model_type)
    elif user_input == "prediction":
        print("prediction process selected")
    else:
        raise Exception("process not defined")