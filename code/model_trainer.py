import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input,  Flatten, Concatenate, GRU
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from attention import Bahdanau, Luong

class Trainer:
    """
    Class for handling training with an LST<. 
    
    Members:
    
     - test_year : year to use as test. The previous 3 are used for training and feature selection
     - feature_creation_space_size : [int] number of feature to create and use
     - small_batch_size : [boolean] if true 32 batch size is used, otherwise 512 (same as paper) 
     - model_type : select model among lstm+soft_attention, lstm+custom_soft_attention, lstm+custom_general_attention and lstm. 
     - epochs : [int] epochs for training

    Functions:

     - setup_lstm : set up keras model 
     - setup_callbacks 
     - reshape : helper used in train to reshape arrays
     - train : train and predict

    """
   

    def __init__(self, test_year, feature_creation_space_size, small_batch_size=False, model_type='lstm', epochs=1000):
    #   ----------------------------------------------------- 
    
        """
        Initialize. Default parameters specified in Pushpendu Ghosh's paper
        """
    #   ----------------------------------------------------- 
    
        self.test_year = test_year
        self.epochs = epochs
        self.model_type = model_type
        print(self.model_type)
        if small_batch_size:
            self.batch_size = 32
        else:
            self.batch_size = 512
        self.feature_creation_space_size = feature_creation_space_size
        if model_type=='lstm+soft_attention' and not small_batch_size:
            print("Attention: large batch size + attention requires a large RAM")
            


    #   ----------------------------------------------------- 
    def setup_lstm(self):
    #   ----------------------------------------------------- 
    
        """
        set up the LSTM model:
         - Input : the 240x3 feature matrix of each stock;
         - LSTM layer with 25 units, tanh activation, sigmoid recurrent_activation;
         - Drop out regularizer layer with 0.1 rate;
         - Dense layer with 2 nodes and softmax activation.
        
        ------
        input  : None
        output : model

        """
    #   ----------------------------------------------------- 
        inputs = Input(shape=(self.feature_creation_space_size,3))
        
        if self.model_type=='lstm+soft_attention':
            x, last_hidden_state, _ = LSTM(10, return_sequences=True, return_state=True)(inputs)
            x = Dropout(0.1)(x)
            x = tf.keras.layers.AdditiveAttention()([x, last_hidden_state])
            x = Flatten()(x)
        
        elif self.model_type=='lstm+custom_soft_attention':
            x, last_hidden_state, _ = LSTM(10, return_sequences=True, return_state=True)(inputs)
            x = Dropout(0.5)(x)
            attention_layer = Bahdanau(5)
            x = attention_layer(x, last_hidden_state)
            x = Flatten()(x)
        
        elif self.model_type=='lstm+custom_general_attention':
            x= LSTM(25, return_sequences=True, return_state=False)(inputs)
            x = Dropout(0.5)(x)
            attention_layer = Luong(25)
            x = attention_layer(x)
            x = Flatten()(x)
        
        elif self.model_type=='lstm':
            x = LSTM(25, return_sequences=False, return_state=False)(inputs)
            x = Dropout(0.1)(x)
        
        else:
            raise ValueError('The possible models are lstm+soft_attention, lstm+custom_soft_attention, lstm+custom_general_attention and lstm.')
        
        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(), metrics=['accuracy'])
        model.summary()
        return model


    #   ----------------------------------------------------- 
    def setup_callbacks(self):
    #   ----------------------------------------------------- 
    
        """
        set up lis of callbacks:
         - EarlyStopping to avoid overfitting, minimising the value of cost function on the cross-validation data,
            a patience of 10 epochs and restoring weights from the best value of the loss
         - ModelCheckpoint with a period of 1
         - CSV logger
        
        ------
        input  : None
        output : callbacks

        """
    #   ----------------------------------------------------- 
    
        callbacks = [
        tf.keras.callbacks.EarlyStopping(mode='min', patience=10, restore_best_weights=True),#patience=10 in original
        tf.keras.callbacks.ModelCheckpoint(filepath='model_'+self.model_type+'/'+str(self.test_year)+"-E{epoch:02d}.h5", save_freq=1),
        tf.keras.callbacks.CSVLogger(filename='model_'+self.model_type+'/'+str(self.test_year)+'_traininglog.csv' ),
        ]

        return callbacks



    #   ----------------------------------------------------- 
    def reshape(self, x):
    #   ----------------------------------------------------- 
        """
        swap axes
        """
        x = np.array(np.split(x, 3, axis=1))
        x = np.swapaxes(x, 0, 1)
        x = np.swapaxes(x, 1, 2)
        return x



    #   ----------------------------------------------------- 
    def train(self, train_data, test_data):
    #   ----------------------------------------------------- 
    
        """
        set up the LSTM model:
        split train data in predictors and dependent component, reshape according to model 
        expected input and using sklearn one-ho-encoder, call the setup methods and fit the model.
        Finally predictions are made fore each day in test set.

        ------
        input  : train_data, test_data
        output : None

        """
    #   ----------------------------------------------------- 
        
        np.random.shuffle(train_data)
        train_x, train_y = train_data[:,2:-2], train_data[:,-1]
        train_x = self.reshape(train_x)
        train_y = np.reshape(train_y,(-1, 1))
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(train_y)
        encoder_y = encoder.transform(train_y).toarray()
    
        model =  self.setup_lstm()
        callbacks = self.setup_callbacks()
        
        train_x = np.asarray(train_x).astype('float32')
        model.fit(train_x,
                  encoder_y,
                  epochs=self.epochs,
                  validation_split=0.2,
                  callbacks=callbacks,
                  batch_size=self.batch_size
                  )
    
        dates = list(set(test_data[:,0]))
        predictions = {}
        for day in dates:
            test_day = test_data[test_data[:,0]==day]
            test_day = self.reshape(test_day[:,2:-2])
            test_day = np.asarray(test_day).astype('float32')
            predictions[day] = model.predict(test_day)[:,1]
        
        return model, predictions



