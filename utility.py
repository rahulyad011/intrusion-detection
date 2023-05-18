import os
import re
import pandas as pd
from os import path

# filepath = "/content/drive/MyDrive/Colab_Notebooks/Datasets/kitsume_network_attack_dataset/ARP_MitM_dataset_kitsune_binary_data_1M.csv"
datadir = "/Users/ryadav704/Projects/NIDS/git_code/intrusion-detection/datasets/"

def find_dataset(datadir_sel, dataset_name):
    print("dataset_name to be searched:", dataset_name)
    file_path = ""
    for dirname, _, filenames in os.walk(datadir_sel):
        print("filenames: ", filenames)
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
        datadir_kit = datadir+"kitsume_network_attack_dataset"
        datafilepath = find_dataset(datadir_kit, data_req_param[1])
    else:
        datafilepath = find_dataset(datadir, data_req_param[0])
        # datafilepath = datadir+"CICIDS.csv"
    if datafilepath == "":
        raise Exception("the requested dataset does not exist in the data directory")
        return 0
    bin_data_total = pd.read_csv(datafilepath)
    bin_data_total.drop(bin_data_total.columns[0],axis=1,inplace=True)
    if "kitsune" == data_req_param[0]:
        bin_data_total.label = bin_data_total.label.astype(int)
        if 'Unnamed: 0' in bin_data_total.columns:
            bin_data_total.drop('Unnamed: 0',axis=1,inplace=True)
    else:
        new_col_name = 'label'
        old_col_name = bin_data_total.columns[-1]
        bin_data_total = bin_data_total.rename(columns={old_col_name: new_col_name})
    print("printing head of the loaded dataset")
    print(bin_data_total.head())
    print_data_statistics(bin_data_total)
    return bin_data_total

def save_eval_in_csv(dataset_used, model_type, c_accuracy, c_f1, c_fnr, c_fpr):
    result_filename = 'results/classfiers_evaluation_metrics.csv'
    if (not path.isfile(result_filename)):
        # Create an empty DataFrame with the desired columns
        df = pd.DataFrame(columns=['Dataset', 'Classifier', 'Accuracy', 'F1_Score', 'False_Negative_Rate',  'False_Postive_Rate'])
        # Save the empty DataFrame to a CSV file
        df.to_csv(result_filename, index=False)
    df = pd.read_csv(result_filename)
    # Check if a row with the same dataset and classifier already exists
    mask = (df['Dataset'] == dataset_used) & (df['Classifier'] == model_type)
    if df.loc[mask].shape[0] > 0:
        # Overwrite the existing row with the new values
        df.loc[mask, 'Accuracy'] = c_accuracy
        df.loc[mask, 'F1_Score'] = c_f1
        df.loc[mask, 'False_Negative_Rate'] = c_fnr
        df.loc[mask, 'False_Postive_Rate'] = c_fpr
        df.to_csv(result_filename, index=False)
        print(f"Updated existing entry: Dataset={dataset_used}, Classifier={model_type}")
    else:
        # Create a new DataFrame with the new entry
        new_entry = pd.DataFrame([[dataset_used, model_type, c_accuracy, c_f1, c_fnr, c_fpr]], columns=['Dataset', 'Classifier', 'Accuracy', 'F1_Score', 'False_Negative_Rate',  'False_Postive_Rate'])
        
        # Concatenate the new DataFrame with the original DataFrame and save to CSV
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(result_filename, index=False)
        print(f"Added new entry: Dataset={dataset_used}, Classifier={model_type}")