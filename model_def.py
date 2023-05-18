# importing library for support vector machine classifier
from sklearn.svm import SVC

# MLP requirements 
from keras.layers import Dense # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.models import model_from_json # saving and loading trained model

# LSTM requirements 
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model

# representation of model layers
from keras.utils.vis_utils import plot_model

def mlp_def(input_layer_dim):
    mlp = Sequential() # creating model

    # adding input layer and first layer with 50 neurons
    # mlp.add(Dense(units=50, input_dim=X_train.shape[1], activation='relu'))
    mlp.add(Dense(units=50, input_dim=input_layer_dim, activation='relu'))
    # output layer with sigmoid activation
    mlp.add(Dense(units=1,activation='sigmoid'))
    # defining loss function, optimizer, metrics and then compiling model
    mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summary of model layers
    print(mlp.summary())
    return mlp

def lstm_def(input_layer_dim):
    lst = Sequential()
    # input layer and LSTM layer with 50 neurons
    input_features = input_layer_dim
    lst.add(LSTM(50,input_dim=input_features))

    # outpute layer with sigmoid activation
    lst.add(Dense(1,activation='sigmoid'))
    # defining loss function, optimizer, metrics and then compiling model
    lst.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    # summary of model layers
    print(lst.summary())
    return lst

def autoencoder_def(input_layer_dim):
    input_dim = input_layer_dim
    encoding_dim = 50

    #input layer
    input_layer = Input(shape=(input_dim, ))
    #encoding layer with 50 neurons
    encoder = Dense(encoding_dim, activation="relu")(input_layer)           
    #decoding and output layer
    output_layer = Dense(input_dim, activation='softmax')(encoder)
    # creating model with input, encoding, decoding, output layers
    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # defining loss function, optimizer, metrics and then compiling model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

    # summary of model layers
    print(autoencoder.summary())
    return autoencoder

def autoencoder_classifier_def(input_layer_dim):    
    i_dim = input_layer_dim
    #input layer
    i_layer = Input(shape=(i_dim, ))
    #hidden layer with 50 neurons
    fvector = Dense(50, activation="sigmoid")(i_layer)                    
    #doutput layer
    o_layer = Dense(1, activation='sigmoid')(fvector)
    # creating model with input, encoding, decoding, output layers
    ae_classifier = Model(inputs=i_layer, outputs=o_layer)

    # defining loss function, optimizer, metrics and then compiling model
    ae_classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    # summary of model layers
    print(ae_classifier.summary())
    return ae_classifier

def select_load_model_def(model_type, input_dim):
    if model_type == "SVM":
        print("load SVM classifier def:")
        lsvm = SVC(kernel='linear',gamma='auto') 
        return lsvm
    elif model_type == "MLP":
        return mlp_def(input_dim)
    elif model_type == "LSTM":
        return lstm_def(input_dim)
    elif model_type == "AE":
        return autoencoder_def(input_dim)
    else:
        return None