# numeric feature normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import numpy as np

def normalization_train(df, col, scaler_obj):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = scaler_obj.fit_transform(arr.reshape(len(arr),1))
  return df

def normalization_test(df, col, scaler_obj):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = scaler_obj.transform(arr.reshape(len(arr),1))
  return df

def preprocessing(df, norm_type, process_type):
    if norm_type == "std_scalar":
        # using standard scaler for normalizing
        scaler = StandardScaler()
    else:
        # using minmax scaler for normalizing
        # to avoid negative values after scaling
        scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(['number']).columns
    # numeric_cols = numeric_cols.drop('label')
    if process_type == "train":
        norm_data = normalization_train(df, numeric_cols, scaler)
    elif process_type == "test":
        norm_data = normalization_test(df, numeric_cols, scaler)
    else:
        raise Exception("preprocessing process not defined")
    return norm_data
