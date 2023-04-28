# for splitting the dataset for training and testing
from sklearn.model_selection import train_test_split
# importing library for support vector machine classifier
from sklearn.svm import SVC
# saving and loading trained model
import pickle
# getting path/dir using OS
from os import path
#numpy
import numpy as np
# for calculating accuracy of model
from sklearn.metrics import accuracy_score

# local imports
from utility import load_dataset, print_data_statistics
from preprocessing import preprocess

#local imports
from model_def import select_load_model_def
from evaluation import evaluate

def save_trained_model(model, model_dir,  model_name_suffix):
    pkl_filename = model_dir+"/"+"binary"+model_name_suffix+".pkl"
    if (not path.isfile(pkl_filename)):
        # saving the trained model to disk
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
            print("Saved model to disk")
    else:
        print("model with the same name exists on the disk, overwriting the existing model file")
        # saving the trained model to disk
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
            print("Saved model to disk")

def load_saved_model(model_dir, model_name_suffix):
    pkl_filename =  model_dir+ "/"+ "binary" + model_name_suffix + ".pkl"
    print("loading model from path:", pkl_filename)
    model_cls = None
    if (path.isfile(pkl_filename)):
        print("inside folder")
        # loading the trained model from disk
        with open(pkl_filename, 'rb') as file:
            model_cls = pickle.load(file)
            print("Loaded model from disk")
    else:
        print("No such saved model exists in the model directory, please train again")
    return model_cls

def data_split(bin_data):
    # splitting the dataset 75% for training and 25% testing
    number_of_cols = bin_data.shape[1]-1
    X = bin_data.iloc[:,0:number_of_cols] # dataset excluding target attribute (encoded, one-hot-encoded,original)
    Y = bin_data['label'] # target attribute
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=42) 
    print("number of rows in X_train")
    print(X_train.shape[0])
    print("number of rows in X_test")
    print(X_test.shape[0])
    return X_train, X_test, y_train, y_test

def train(data_params, model_type, predict_only):
    print("data_param: ", data_params)
    data = load_dataset(data_params)
    print("data is loaded for model training")
    print("spliting the dataset in train and test:")
    X_train, X_test, y_train, y_test = data_split(data)
    print("normalize the numeric columns")
    X_train_norm, X_test_norm = preprocess(X_train, X_test, "min_max")
    print("X_train_norm shape:", X_train_norm.shape)
    print("X_train_norm shape:", X_test_norm.shape)
    # select and load model definition
    model = select_load_model_def(model_type)
    model_save_dir = "Models"
    #model naming conventions
    data_name_arr = data_params.split()
    data_string = "_".join(data_name_arr)
    model_suffix_name = model_type+data_string
    # model training:
    if model_type == "SVM":
        if predict_only:
            # load model
            print("loading model for prediction: ")
            class_model = load_saved_model(model_save_dir, model_suffix_name)
            print("checking loaded model: ")
            if class_model is None:
                raise Exception("requested model doesn't exist, please train again")
            predict(class_model, model_type, X_test_norm, y_test)
        else:
            print("start model training:")
            model.fit(X_train_norm,y_train) # training model on training dataset
            save_trained_model(model, model_save_dir, model_suffix_name)
            predict(model, data_string, model_type, X_test_norm, y_test)
    elif model_type == "MLP":
        history = model.fit(X_train, y_train, epochs=100, batch_size=5000,validation_split=0.2)
    elif model_type == "LSTM":
        X_train = np.array(X_train)
        x_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
        history = model.fit(x_train, y_train, epochs=100, batch_size=5000,validation_split=0.2)
    elif model_type == "AE":
        pass
    else:
        raise Exception("requested model doesn't exist")
        return None
    # print(X_test.head())

def predict(classifier, dataset_name, model_type, x_test_data, y_test_data):
    y_pred = classifier.predict(x_test_data) # predicting target attribute on testing dataset
    ac = accuracy_score(y_test_data, y_pred)*100 # calculating accuracy of predicted data
    print("Classifier Binary Accuracy is ", ac)
    plot_required = True
    print("Evaluating the Classifier:")
    evaluate(classifier, dataset_name, model_type, y_test_data, y_pred, plot_required)


