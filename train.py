import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization


"""
Parameters to tune
"""
softness = 0 # range (0,1)
number_of_units = 100
learning_rate = 0.002 # initial learning rate of optimizer
patience = 42 # number of epochs with no improvement after which training will be stopped
equipment_prediction_error_function = 'categorical_crossentropy'
time_prediction_error_function = 'mae'
validation_split = 0.2
epochs = 500
"""
Other parameters (no need to change)
"""
initializer = 'glorot_uniform' # initialize unit

# Train the model using vectorized features

def train(X: [[[]]], equipment_prediction: [[]], time_prediction: [], max_case_len: int, models_path: str, num_features: int, number_of_eids: int):
    print('Build model...')
    # Define the main input layer
    main_input = Input(shape=(max_case_len, num_features), name='main_input')

    # Shared LSTM layer
    l1 = LSTM(number_of_units, implementation=2, kernel_initializer=initializer, return_sequences=True, dropout=0.2)(main_input)
    b1 = BatchNormalization()(l1)

    # LSTM layer specialized in EID prediction
    l2_eid = LSTM(number_of_units, implementation=2, kernel_initializer=initializer, return_sequences=False, dropout=0.2)(b1)
    b2_eid = BatchNormalization()(l2_eid)

    # LSTM layer specialized in dwell time prediction
    l2_time = LSTM(number_of_units, implementation=2, kernel_initializer=initializer, return_sequences=False, dropout=0.2)(b1)
    b2_time = BatchNormalization()(l2_time)

    # Output layers for EID and dwell time prediction
    equipment_output = Dense(number_of_eids, activation='softmax', kernel_initializer=initializer, name='equipment_output')(b2_eid)
    time_output = Dense(1, kernel_initializer=initializer, name='time_output')(b2_time)

    # Create the model
    model = Model(inputs=[main_input], outputs=[equipment_output, time_output])

    # Define the optimizer (Adam)
    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    # Compile the model with loss functions and optimizer
    model.compile(loss={'equipment_output': equipment_prediction_error_function,'time_output': time_prediction_error_function}, optimizer=opt)

    # Define the callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model_checkpoint = ModelCheckpoint(models_path+'model_{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=False, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    # Train the model     
    model.fit(X, {'equipment_output':equipment_prediction, 'time_output':time_prediction},validation_split=validation_split, verbose=2,callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=max_case_len, epochs=epochs)





