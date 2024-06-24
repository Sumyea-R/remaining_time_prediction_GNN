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
import constants
import routing_encoding


def encode_state(event: Event, eid_mapper: bidict, idx: int) -> [int]:
    """
    One-hot encoding for the current state
    """

    # Create an empty list to represent the one-hot encoded state
    eid_vector = [0]*(len(eid_mapper)-1)

    # Check if the EID of the current event is in the EID mapper
    if event[constants.attr_equipment] in eid_mapper:
        id_of_eid = eid_mapper[event[constants.attr_equipment]]

        # Set the corresponding element in the one-hot encoded state to 1 (as true)
        eid_vector[id_of_eid-1] = 1 

        # Pointer of the added state corresponding to a case
        eid_vector.append(idx + 1)
    else:
        eid_vector.append(0) # If the EID is not in the mapper, set the element to 0

    return eid_vector


def encode_time_related_features(event: Event, prev_event: Event, case_start_time: datetime, average_dwell_time_in_sec: float, average_time_since_start_in_sec):
    """
    Encode features related to time
    """
    # normalized dwell time of current event
    dwell_time = event[constants.attr_arrived_time] - prev_event[constants.attr_arrived_time] if prev_event is not None else event[constants.attr_arrived_time] - case_start_time
    #average_dwell_time_of_eqtyp = dwell_time_mapper[event['EQTYP']]
    normalized_dwell_time = dwell_time.total_seconds()/average_dwell_time_in_sec
    #normalized_dwell_time = dwell_time.total_seconds()/average_dwell_time_of_eqtyp

    # normalized time of current event since case starts
    time_since_case_start =  event[constants.attr_arrived_time] - case_start_time
    normalized_time_since_case_start = time_since_case_start.total_seconds()/average_time_since_start_in_sec
    
    # time since midnight (normalized)
    time_since_midnight_in_seconds = (event[constants.attr_arrived_time] - event[constants.attr_arrived_time].replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    normalized_time_since_midnight = time_since_midnight_in_seconds/86400

    # the weekday of event (normalized)
    weekday = event[constants.attr_arrived_time].weekday()
    normalized_weekday = event[constants.attr_arrived_time].weekday()/7
    
    # Encode whether it's a weekend or not
    is_weekend = 1 if weekday in [5, 6] else 0
    
    return [normalized_dwell_time, normalized_time_since_case_start, normalized_time_since_midnight, normalized_weekday, is_weekend]


def encode_event_attribute_features(event: Event, event_attribute_mappers: dict) -> [int]:
    """
    Encode the event attributes into a numerical feature vector.
    """
    attributes = sorted(list(event_attribute_mappers.keys()))
    event_attr_vector = []
    last_four_digits = ''.join(event[constants.attr_equipment][4:])
    for attr, mapper in event_attribute_mappers.items():
        if event[attr] in mapper:
            event_attr_vector.append(mapper[event[attr]]) # Encode event attributes w. mapper
        else:
            event_attr_vector.append(0) # Set to 0 if the attribute is not in the mapper
    event_attr_vector.append(int(last_four_digits)) # Add the last four digits of EID as feature
    return event_attr_vector


def encode_case_attribute_features(case: Trace, case_attribute_mappers: dict, eid_mapper: bidict) -> [int]:
    """
    Encode the case attributes into a numerical feature vector.
    """
    attributes = sorted(list(case_attribute_mappers.keys()))
    case_attr_vector = []
    for attr in constants.case_attribute_features:
        if case.attributes[attr] in eid_mapper:
            case_attr_vector.append(eid_mapper[case.attributes[attr]]) # Encode the case attribute w. mapper
        else:
            case_attr_vector.append(0) # Set to 0 if the attribute is not in the mapper  
    return case_attr_vector


def encode_state_features(case_state: [Event], case: Trace, average_time_since_start_in_sec: float, eid_mapper: bidict, average_dwell_time_in_sec: float, event_attribute_mappers: dict, case_attribute_mappers:dict, network: nx.DiGraph) :
    """
    Encode the features for each state in the case state
    """

    # Initialize an empty list to store the encoded state features
    state_feature = []
    edge_weight = []
    edge_idx = []
    events_in_prefix = []

    # Iterate over each state in the case state
    for idx, event in enumerate(case_state):
        
        events_in_prefix.append(event[constants.attr_equipment])

        # Initialize an empty list to store the feature vector for the event
        feature_vector = []        

        # Encode the current event and add it to the feature vector
        feature_vector += encode_state(event, eid_mapper, idx)
        
        # Encode time-related features and add them to the feature vector
        prev_event = case_state[idx-1] if idx >= 1 else None 
        time_features = encode_time_related_features(event, prev_event, case.attributes['Init Datetime'], average_dwell_time_in_sec, average_time_since_start_in_sec)
        feature_vector += time_features

        # Encode event attribute features and add them to the feature vector
        feature_vector += encode_event_attribute_features(event, event_attribute_mappers)

        # Encode case attribute features and add them to the feature vector
        feature_vector += encode_case_attribute_features(case, case_attribute_mappers, eid_mapper)
        
        # Encode routing information and add them to the feature vector
        #feature_vector += routing_encoding.encode_routing_features(eid_mapper, case, event, network, 2)
        
        # Add edge weight and index
        if idx < len(case_state) - 1 :
            if event[constants.attr_equipment] == case_state[idx+1][constants.attr_equipment] :
                weight = 0
            elif case_state[idx+1][constants.attr_equipment] in events_in_prefix :
                weight = -1
            else :
                weight = 1
                
            edge_weight.append(weight)
            edge_idx.append([idx, idx+1])
        
        # Append the feature vector to the state_feature list
        state_feature.append(feature_vector)

    # Return the encoded vectors for every event from the input (case_state)
    return [state_feature, edge_idx, edge_weight]




