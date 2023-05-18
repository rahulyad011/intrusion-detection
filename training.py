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
#pandas
import pandas as pd
# for calculating accuracy of model
from sklearn.metrics import accuracy_score

# local imports
from utility import load_dataset, print_data_statistics
from preprocessing import preprocess

#keras
from keras.models import model_from_json # saving and loading trained model

#local imports
from model_def import select_load_model_def
from model_def import autoencoder_classifier_def
from evaluation import evaluate

def save_trained_model(model, model_dir,  model_name_suffix, model_type):
    if model_type=="SVM":
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
    else:
        filepath = model_dir+"/"+"binary"+model_name_suffix+'.json'
        weightspath = model_dir+"/"+"binary"+model_name_suffix+'.h5'
        if (not path.isfile(filepath)):
            # serialize model to JSON
            model_json = model.to_json()
            with open(filepath, "w") as json_file:
                json_file.write(model_json)

            # serialize weights to HDF5
            model.save_weights(weightspath)
            print("Saved model to disk")

def load_saved_model(model_dir, model_name_suffix, model_type):
    model_cls = None
    if model_type=="SVM":
        pkl_filename =  model_dir+ "/"+ "binary" + model_name_suffix + ".pkl"
        print("loading model from path:", pkl_filename)
        if (path.isfile(pkl_filename)):
            print("inside folder")
            # loading the trained model from disk
            with open(pkl_filename, 'rb') as file:
                model_cls = pickle.load(file)
                print("Loaded model from disk")
        else:
            print("No such saved model exists in the model directory, please train again")
    else:
        # pkl_filename =  model_dir+ "/"+ "binary" + model_name_suffix + ".pkl"
        filepath = model_dir+"/"+"binary"+model_name_suffix+'.json'
        weightspath = model_dir+"/"+"binary"+model_name_suffix+'.h5'
        print("loading model file from path:", filepath)
        print("loading weight file from path:", weightspath)
        # load json and create model
        json_file = open(filepath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_cls = model_from_json(loaded_model_json)
        # load weights into new model
        model_cls.load_weights(weightspath)
        print("Loaded model from disk")
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

def train(data_params, model_type, predict_only, label_field):
    print("data_param: ", data_params)
    data = load_dataset(data_params)
    print("data is loaded for model training")
    print("spliting the dataset in train and test:")
    X_train, X_test, y_train, y_test = data_split(data)
    print("normalize the numeric columns")
    X_train_norm, X_test_norm = preprocess(X_train, X_test, "min_max")
    print("X_train_norm shape:", X_train_norm.shape)
    print("X_test_norm shape:", X_test_norm.shape)
    # select and load model definition
    model = select_load_model_def(model_type, X_train_norm.shape[1])
    model_save_dir = "Models"
    #model naming conventions
    data_name_arr = data_params.split()
    data_string = "_".join(data_name_arr)
    model_suffix_name = model_type+data_string
    # model training:
    if predict_only:
        # load model
        print("loading model for prediction: ")
        class_model = load_saved_model(model_save_dir, model_suffix_name, model_type)
        print("checking loaded model: ")
        if class_model is None:
            raise Exception("requested model doesn't exist, please train again")
        predict_test(class_model, data_string, model_type, X_test_norm, y_test, model_suffix_name)
    else:
        print("start model training:")
        if model_type == "SVM":
            model.fit(X_train_norm,y_train) # training model on training dataset
        elif model_type == "MLP":
            history = model.fit(X_train_norm, y_train, epochs=5, batch_size=64,validation_split=0.2)
        elif model_type == "LSTM":
            X_train_norm = np.array(X_train_norm)
            X_train_norm = np.reshape(X_train_norm, (X_train_norm.shape[0],1,X_train_norm.shape[1]))
            history = model.fit(X_train_norm, y_train, epochs=5, batch_size=64,validation_split=0.2)
        elif model_type == "AE":
            # dataset excluding target attribute (encoded, one-hot-encoded,original)
            print(X_test_norm.columns)
            print("label col:", label_field)
            X_train_norm = X_train_norm.values
            X_test = X_test.values
            y_test = y_test.values
            history = model.fit(X_train_norm, X_train_norm, epochs=5, batch_size=64,validation_data=(X_test_norm, X_test_norm)).history
        else:
            raise Exception("requested model doesn't exist")
            return None
        print("start model saving")
        save_trained_model(model, model_save_dir, model_suffix_name, model_type)
        print("start prediction on test data")
        predict_test(model, data_string, model_type, X_test_norm, y_test, model_suffix_name)
    # print(X_test.head())

def ae_classifier(predictions_data, model_type, dataset_name, x_test_data, y_test, model_suffix_name):
    i_dim = predictions_data.shape[1]
    ae_model = autoencoder_classifier_def(i_dim)
    model_save_dir = "Models"
    his = ae_model.fit(predictions_data, y_test, epochs=5, batch_size=50, validation_split=0.2).history
    print("start model saving")
    save_trained_model(ae_model, model_save_dir, model_suffix_name, "AE_Clas")
    print("start prediction on test data")
    test_results = ae_model.evaluate(x_test_data, y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}')
    # y_pred = classifier.predict(x_test_data).ravel()
    y_pred = (ae_model.predict(x_test_data)>0.5).astype('int32')
    plot_required = True
    print("Evaluating the Classifier:")
    evaluate(ae_model, dataset_name, model_type, y_test, y_pred, plot_required)
    return 0

def predict_test(classifier, dataset_name, model_type, x_test_data, y_test_data, model_suffix_name):
    if model_type == "SVM":
        y_pred = classifier.predict(x_test_data) # predicting target attribute on testing dataset
        ac = accuracy_score(y_test_data, y_pred)*100 # calculating accuracy of predicted data
        print("Classifier Binary Accuracy is ", ac)
        plot_required = True
        print("Evaluating the Classifier:")
        evaluate(classifier, dataset_name, model_type, y_test_data, y_pred, plot_required)
    else:
        if model_type == "LSTM":
            x_test_data = np.reshape(x_test_data, (x_test_data.shape[0],1,x_test_data.shape[1]))
            print("shape of x_test_data", x_test_data.shape)
        classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if model_type == "AE":
            test_results = classifier.evaluate(x_test_data, x_test_data, verbose=1)
            predictions = classifier.predict(x_test_data)
            mse = np.mean(np.power(x_test_data - predictions, 2), axis=1)
            error_df = pd.DataFrame({'reconstruction_error': mse,
                                    'true_class': y_test_data})
            ae_classifier(predictions, model_type, dataset_name, x_test_data, y_test_data, model_suffix_name)
            # print("to be implemented")
        else:
            test_results = classifier.evaluate(x_test_data, y_test_data, verbose=1)
            print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}')
            # y_pred = classifier.predict(x_test_data).ravel()
            y_pred = (classifier.predict(x_test_data)>0.5).astype('int32')
            plot_required = True
            print("Evaluating the Classifier:")
            evaluate(classifier, dataset_name, model_type, y_test_data, y_pred, plot_required)

# for later use
def predict_single(classifier, dataset_name, model_type, test_record):
    y_pred = classifier.predict(test_record) # predicting target attribute on testing dataset
    return y_pred
