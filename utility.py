import os
import re
import pandas as pd

# filepath = "/content/drive/MyDrive/Colab_Notebooks/Datasets/kitsume_network_attack_dataset/ARP_MitM_dataset_kitsune_binary_data_1M.csv"
datadir = "/Users/ryadav704/Projects/NIDS/API_code/datasets"

def find_dataset(datadir_sel, dataset_name):
    file_path = ""
    for dirname, _, filenames in os.walk(datadir_sel):
        for filename in filenames:
            # re.search("^The.*Spain$", txt)
            if dataset_name in filename:
                print("current file:", filename)
                print("dir file:", dirname)
                file_path = os.path.join(datadir_sel, filename)
                print("found file path:", file_path)
                break
    return file_path

def print_data_statistics(df):
    stats= {}
    # distribution of label class before sampling
    number_of_rows = df.shape[0]
    stats["nrows"] = number_of_rows
    # exclude label from the feature count
    number_of_feature = df.shape[1]-1
    stats["ncols"] = number_of_rows
    # label split
    label_counts = dict(df.label.value_counts()/number_of_rows)
    stats["split"] = label_counts
    print(stats)
    return stats

def load_dataset(dataset_req):
    data_req_param = dataset_req.split()
    if "kitsune" == data_req_param[0]:
        datadir_kit = datadir+"/kitsume_network_attack_dataset"
        datafilepath = find_dataset(datadir_kit, data_req_param[1])
    else:
        datafilepath = find_dataset(datadir, data_req_param[0])
    if datafilepath == "":
        raise Exception("the requested dataset does not exist in the data directory")
        return 0
    bin_data_total = pd.read_csv(datafilepath)
    bin_data_total.drop(bin_data_total.columns[0],axis=1,inplace=True)
    if "kitsune" == data_req_param[0]:
        bin_data_total.label = bin_data_total.label.astype(int)
        bin_data_total.drop('Unnamed: 0',axis=1,inplace=True)
    print("printing head of the loaded dataset")
    print(bin_data_total.head())
    print_data_statistics(bin_data_total)
    return bin_data_total