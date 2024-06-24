import pandas as pd
import pm4py
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
import feature_encoding
import utility_functions
import constants


def predict(log: EventLog, model: Model, average_time_since_start_in_sec: float, eid_mapper: bidict, dwell_time_mapper:bidict, event_attribute_mappers: dict, case_attribute_mappers:dict, max_case_len: int, network: nx.DiGraph, starting_prefix_size: int = 2) -> [[str, str, datetime]]:
    """
    Prediction using trained LSTM model
    """
    all_prediction = []
    
    for case in log:
        
        events_selected = case[:starting_prefix_size] # Prediction from starting_prefix_size+1
              
        for prefix_size in range(starting_prefix_size, max_case_len):
            last_state =  feature_encoding.encode_state_features(events_selected, case, average_time_since_start_in_sec, eid_mapper, dwell_time_mapper, event_attribute_mappers, case_attribute_mappers, network)
            features = utility_functions.format_input_features([last_state], max_case_len)
            prediction = model.predict(features, verbose=0)
            equipment_prediction = prediction[0][0] # Array of probabilities of the next EID
            time_prediction = prediction[1][0][0] # Predicted normalized timestamp

            id_of_equipment_with_highest_probability = np.where(equipment_prediction == np.max(equipment_prediction))[0][0] # EID with highest probability
            next_equipment = eid_mapper.inverse[id_of_equipment_with_highest_probability] # Translate back to actual EID
            
            if(next_equipment == '!'):
                print('final equipment predicted, end case')
                break
            
            next_equipment_eqtyp = ''.join(next_equipment[0:2])
            next_equipment_zone_name = next_equipment[3]
            next_equipment_floor = float(next_equipment[2])
            average_dwell_time_of_eqtyp = dwell_time_mapper[next_equipment_eqtyp]
            
            if time_prediction < 0:
                time_prediction = 0
            
            time_prediction = time_prediction * average_dwell_time_of_eqtyp # De-normalize dwell time prediction
            #time_prediction = time_prediction * average_dwell_time_in_sec
            next_timestamp = events_selected[-1][constants.attr_arrived_time] + timedelta(seconds = time_prediction)
            
            predicted_event = Event({constants.attr_equipment: next_equipment, constants.attr_arrived_time: next_timestamp, 
                                     'EQTYP': next_equipment_eqtyp, 'Zone Name': next_equipment_zone_name, 
                                     'Floor':next_equipment_floor,})
            events_selected.append(predicted_event)
            
            all_prediction.append([case.attributes['concept:name'], next_equipment, next_timestamp])
    
    return all_prediction





