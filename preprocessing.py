# numeric feature normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import numpy as np

def normalization_train(df_train, df_test, col_train, col_test, scaler_obj):
  for i in col_train:
    arr = df_train[i]
    arr = np.array(arr)
    df_train[i] = scaler_obj.fit_transform(arr.reshape(len(arr),1))
    df_test_norm = normalization_test(df_test, col_test, scaler_obj)
  return df_train, df_test_norm

def normalization_test(df, col_test, scaler_obj):
  for i in col_test:
    arr = df[i]
    arr = np.array(arr)
    df[i] = scaler_obj.transform(arr.reshape(len(arr),1))
  return df

def preprocess(train_df, test_df, norm_type):
    if norm_type == "std_scalar":
        # using standard scaler for normalizing
        scaler = StandardScaler()
    else:
        # using minmax scaler for normalizing
        # to avoid negative values after scaling
        scaler = MinMaxScaler()
    numeric_cols_train = train_df.select_dtypes(['number']).columns
    numeric_cols_test = test_df.select_dtypes(['number']).columns
    # numeric_cols = numeric_cols.drop('label')
    norm_data_train, norm_data_test = normalization_train(train_df, test_df, numeric_cols_train, numeric_cols_test, scaler)
    return norm_data_train, norm_data_test
