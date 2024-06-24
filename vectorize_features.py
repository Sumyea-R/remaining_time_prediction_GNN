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
import networkx as nx
import utility_functions
import feature_encoding
import constants
import torch


"""
Parameters to tune
"""
softness = 0 # range (0,1)

def vectorize_features(log: EventLog, average_time_since_start_in_sec: float, eid_mapper: bidict, average_dwell_time_in_sec: float, event_attribute_mappers: dict, case_attribute_mappers:dict, max_case_len: int, network: nx.DiGraph, average_time_till_case_end: float) :
    """
    Vectorize the extracted features for each case in the event log
    """

    # Initialize an lists to store the vectorized features and edge informations
    X = []
    list_edge_idx = []
    list_edge_weight = []

    # Iterate over each case in the event log
    for case in log:
        # Iterate over each event in the case, starting from the second event
        for current_idx, event in enumerate(case[2:], 2):
            # Encode state features for each prefix in the case and append them to X
            state_with_features = feature_encoding.encode_state_features(case[:current_idx], case, average_time_since_start_in_sec, eid_mapper, average_dwell_time_in_sec, event_attribute_mappers, case_attribute_mappers, network)
            #if len(state_with_features[1]) > 0 :
            X.append(torch.tensor(state_with_features[0]))
            list_edge_idx.append(torch.tensor(state_with_features[1]).t().to(torch.long))
            list_edge_weight.append(torch.tensor(state_with_features[2]).t())

        # Encode state features for the entire case (last state) and append them to X
        last_state_with_features = feature_encoding.encode_state_features(case, case, average_time_since_start_in_sec, eid_mapper, average_dwell_time_in_sec, event_attribute_mappers, case_attribute_mappers, network)
        
        X.append(torch.tensor(last_state_with_features[0]))
        list_edge_idx.append(torch.tensor(last_state_with_features[1]).t().to(torch.long))
        list_edge_weight.append(torch.tensor(last_state_with_features[2]).t())

    # Format the input features as a numpy array and return it
    #return utility_functions.format_input_features(X, max_case_len)
    return [X, list_edge_idx, list_edge_weight]

def vectorize_equipment_prediction(log: EventLog, eid_mapper:bidict) -> [[float]]:
    """
    Vectorize EID prediction for each case in the event log
    """

    # List to store the vectorized EID predictions
    equipment_prediction_vector = []

    for case in log:
        for current_idx, event in enumerate(case[:-1]):

            # Create a vector for EID filled with zeros
            equipment_vector = [0]*len(eid_mapper)

            next_equipment = case[current_idx+1][constants.attr_equipment]

            # Set the corresponding index in the vector
            equipment_vector[eid_mapper[next_equipment]] = 1-softness

            equipment_prediction_vector.append(equipment_vector)

        # Create a vector for the last event
        equipment_vector = [0]*len(eid_mapper)

        # Set the index corresponding to '!' (last EID)
        equipment_vector[eid_mapper['!']] = 1-softness

        equipment_prediction_vector.append(equipment_vector)
    
    # Convert to numpy array
    equipment_prediction_vector = np.array(equipment_prediction_vector, dtype = np.float32)
    return equipment_prediction_vector


def vectorize_time_prediction(log: EventLog, dwell_time_mapper: bidict) -> [float]:
    """
    Vectorize dwell time prediction for each case in the event log
    """

    # List to store the vectorized time predictions
    time_prediction_vector = []

    for case in log:
        for current_idx, event in enumerate(case[:-1]):

            # Calculate the difference of timestamp for dwell time
            predicted_dwell_time = case[current_idx+1][constants.attr_arrived_time] - event[constants.attr_arrived_time]

            # Normalize the dwell time prediction
            average_dwell_time_of_eqtyp = dwell_time_mapper[case[current_idx+1]['EQTYP']]
            normalized_time_prediction = predicted_dwell_time.total_seconds() / average_dwell_time_of_eqtyp
            #normalized_time_prediction = predicted_dwell_time.total_seconds() / average_dwell_time_in_sec

            # Append the normalized dwell time prediction
            time_prediction_vector.append(normalized_time_prediction)

        # Append 0 seconds for the last event
        time_prediction_vector.append(timedelta(seconds=0).total_seconds())

    time_prediction_vector = np.array(time_prediction_vector, dtype=np.float32)
    return time_prediction_vector

def vectorize_rtm_prediction(log: EventLog, average_time_till_case_end: float) -> []:
    """
    Vectorize dwell time prediction for each case in the event log
    """

    # List to store the vectorized time predictions
    time_prediction_vector = []

    for case in log:
        for current_idx, event in enumerate(case[1:-1]):

            # Calculate the difference of timestamp for dwell time
            predicted_dwell_time = case.attributes['Complete Datetime'] - event[constants.attr_arrived_time]

            # Normalize the remaining time prediction
            normalized_time_prediction = predicted_dwell_time.total_seconds() / average_time_till_case_end

            # Append the normalized dwell time prediction
            time_prediction_vector.append(normalized_time_prediction)

        # Append 0 seconds for the last event
        time_prediction_vector.append(timedelta(seconds=0).total_seconds())

    time_prediction_vector = torch.tensor(time_prediction_vector)
    return time_prediction_vector



