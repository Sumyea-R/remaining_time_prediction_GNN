import pandas as pd
#from pm4py import pm4py
from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import EventLog, Trace, Event
import math
import csv
from bidict import bidict
from datetime import datetime, timedelta
import numpy as np
import itertools
import networkx as nx
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
import pre_processing
import utility_functions
import mappers
import vectorize_features
import train
import prediction
import post_processing
import evaluation
import constants
import preprocess_graph_attributes

# Load and convert the dataset 


# Import the required attributes from the CSV file
df = pd.read_csv('.//toy_filtered.csv')[constants.required_attributes+constants.event_attribute_features+constants.case_attribute]

# Convert the DataFrame into the event log structure
log = xes_converter.apply(df, variant=xes_converter.Variants.TO_EVENT_LOG) 

# Convert the timestamp from string to datetime format
log = pre_processing.convert_timestamp(log)

# Adjust the arrived time
log = pre_processing.adjust_arrived_time(log)


"""
global variables using the entire event log
"""
# Calculate the average dwell time in seconds: 245.68552766191863
average_dwell_time_in_sec = utility_functions.average_dwell_time(log)

# Calculate the average time since case start in seconds: 5737.639253954513
average_time_since_start_in_sec = utility_functions.average_time_since_case_start(log)

# Build the mapper for average dwell time based on EQTYP: 
dwell_time_mapper = mappers.build_average_dwell_time_mapper(log)


"""
Fold creation
"""
# Create folds for training and testing
folds = pre_processing.create_folds(log, 3)

# get the training log from first two folds
training_log = EventLog([case for sub_log in folds[:2] for case in sub_log])

# Use the third fold for testing
testing_log = EventLog([case for sub_log in folds[2:] for case in sub_log])


"""
Global variables using only the training log
"""
# Build the mapper for EID based on the training log
eid_mapper = mappers.build_eid_mapper(training_log)

# Get the maximum case length in the training log and add an end signal
max_case_len = max([len(case) for case in training_log]) + 1

# Build mappers for event attribute features based on the training log
event_attribute_mappers = {attr: mappers.build_event_attr_mapper(training_log, attr) for attr in constants.event_attribute_features}

# Build mappers for case attribute features based on the training log
case_attribute_mappers = {attr: mappers.build_case_attr_mapper(training_log, attr) for attr in constants.case_attribute_features}

# Get the routing netwrok of logistic log
network = preprocess_graph_attributes.preprocess_graph_attributes('.//20230906-logistic_nw.gml')

# Train a model with training log


# Vectorize the features for training
X = vectorize_features.vectorize_features(training_log, average_time_since_start_in_sec, eid_mapper, dwell_time_mapper, event_attribute_mappers, case_attribute_mappers, max_case_len, network)

# Vectorize the expected EID predictions for training
equipment_prediction = vectorize_features.vectorize_equipment_prediction(training_log, eid_mapper) 

# Vectorize the expected dwell time predictions for training
time_prediction = vectorize_features.vectorize_time_prediction(training_log, dwell_time_mapper) 

# Train the model using the vectorized features and predictions
train.train(X, equipment_prediction, time_prediction, max_case_len, constants.models_path, num_features=len(X[0][0]),number_of_eids=len(eid_mapper))


# # Test the model with testing log


# from keras.models import load_model
# # Load the trained model
# model = load_model(constants.models_path+'model_60-5.16.h5')

# # Set the starting prefix size for prediction
# starting_prefix_size = 2

# # Perform predictions on the testing log using the loaded model
# all_prediction = prediction.predict(testing_log, model, average_time_since_start_in_sec, eid_mapper, dwell_time_mapper, event_attribute_mappers, case_attribute_mappers, max_case_len, starting_prefix_size)



# # Create a prefix log based on the testing log with a specified prefix size
# prefix_log = post_processing.create_prefix_log(testing_log, starting_prefix_size)

# # Perform post-processing on the prefix log using the predicted results
# prediction_log = post_processing.post_process(prefix_log, all_prediction)

# # Add ground truth information to the predicted events in the prediction log
# prediction_log = post_processing.add_ground_truth(testing_log, prediction_log)

# # Convert the prediction log to a DataFrame and save it to a CSV file
# result_with_ground_truth = xes_converter.apply(prediction_log, variant=xes_converter.Variants.TO_DATA_FRAME)
# pm4py.write_xes(log, 'result_log.xes')
# result_with_ground_truth.to_csv('result_df.csv')
# # Calculate the accuracy of EID prediction
# accuracy_of_equipment_prediction = evaluation.calculate_accuracy(result_with_ground_truth, len(prediction_log), starting_prefix_size)

# # Calculate MAE and MAPE for dwell time prediction
# error_mappers = evaluation.calculate_mae_mape(prediction_log, starting_prefix_size)

# # Print the MAE and MAPE for timestamp prediction
# print("MAE for next timestamp prediction")
# print(error_mappers['MAE'])
# print("MAPE for next timestamp prediction")
# print(error_mappers['MAPE'])





