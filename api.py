import numpy as np
import pandas as pd

# import local scripts
from training import train

if __name__ == '__main__':
    # user_input = input("which process: ")
    user_input = 't'
    # data = input("which dataset: ")
    # data = 'kitsune SSL'
    # kitsune_datasets_available = ['kitsune Active_Wiretap', 'kitsune SSDP_Flood', 'kitsune ARP_MitM', 'kitsune Video_Injection', 
    # 'kitsune SYN_DoS', 'kitsune SSL_Renegotiation', 'kitsune 1MFuzzing', 'kitsune 1MMirai', 'kitsune 1MOS_Scan']
    datasets_available = ['KDD', 'CICIDS', 'UNSW']
    # models_available = ["MLP", "LSTM", "AE"]
    models_available = ["SVM"]

    data = ""
    model_type = ""
    #for running on all the datasets with all models
    for dataset in datasets_available:
        data = dataset
        for model in models_available:
            model_type = model
            # data = 'CICIDS'
            # model_type = input("which model: ") 
            label_col = ['label']
            if user_input == "t":
                print("train process selected")
                train(data, model_type, False, label_col)
            elif user_input == "p":
                print("predict process selected")
                train(data, model_type, True, label_col)
            else:
                raise Exception("process not defined")