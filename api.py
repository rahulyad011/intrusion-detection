import numpy as np
import pandas as pd

# import local scripts
from training import train

if __name__ == '__main__':
    user_input = input("which process: ")
    data = input("which dataset: ")
    model_type = input("which model: ") 
    if user_input == "t":
        print("train process selected")
        train(data, model_type, False)
    elif user_input == "p":
        print("predict process selected")
        train(data, model_type, True)
    else:
        raise Exception("process not defined")