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
import constants


def build_eid_mapper(log: EventLog) -> bidict:
    """
    Create a mapper between Equipment ID (string) and numbers
    """
    # Get all unique EIDs from the log
    unique_equipment_ids = list(attributes_filter.get_attribute_values(log, constants.attr_equipment).keys())

    # Sort the EIDs in ascending order (this step is not necessary)
    equipment_ids = sorted(unique_equipment_ids) 

    # Create a bidirectional mapper between EID and numbers
    eid_mapper = bidict({equipment_id: idx for idx, equipment_id in enumerate(equipment_ids, 1)})

    # Add a special character '!' mapped to number 0 for ending prediction
    eid_mapper.update({'!': 0})

    return eid_mapper


def build_event_attr_mapper(log: EventLog, event_attribute: str) -> bidict:
    """
    Create a bidirectional mapper between event attribute (string) and numbers.
    """
    # Obtain the unique attribute values for the specified event attribute
    unique_attribute_values = sorted(list(attributes_filter.get_attribute_values(log, event_attribute).keys()))
    # Build the mapper by assigning a unique number to each attribute value
    return bidict({attr_value: idx for idx, attr_value in enumerate(unique_attribute_values, 0)})


def build_case_attr_mapper(log: EventLog, case_attribute: str) -> bidict:
    """
    Create a bidirectional mapper between case attribute (string) and numbers.
    """
    # Obtain the unique attribute values for the specified case attribute
    unique_attribute_values = sorted(list(attributes_filter.get_attribute_values(log, case_attribute).keys()))
    # Build the mapper by assigning a unique number to each attribute value
    return bidict({attr_value: idx for idx, attr_value in enumerate(unique_attribute_values, 0)})


def build_average_dwell_time_mapper(log: EventLog) -> bidict:
    """
    Create a mapper for average dwell time based on eqtyp
    """
    unique_eqtyp = list(attributes_filter.get_attribute_values(log, 'EQTYP').keys())
    
    # Create a bidirectional mapper between EQTYP and its index
    eqtyp_mapper = bidict({attr_value: idx for idx, attr_value in enumerate(unique_eqtyp, 0)})
    
    dwell_times = []# List to store dwell times per EQTYP
    average_dwell_time_per_eqtyp = [] # List to store average dwell time per EQTYP
    
    for i in range(len(unique_eqtyp)): # Initialize lists per EQTYP
        dwell_times.append([])
    
    for case in log:
        # Calculate the dwell time from the case start to the first event's arrival time
        dwell_time = case[0][constants.attr_arrived_time] - case.attributes['Init Datetime']
        eqtyp_idx = eqtyp_mapper[case[0]['EQTYP']]
        dwell_times[eqtyp_idx].append(dwell_time.total_seconds())
        
        # Iterate through each event in the case starting from the second event
        for idx, event in enumerate(case[1:], 1):
            eqtyp_idx = eqtyp_mapper[event['EQTYP']]
            dwell_time = event[constants.attr_arrived_time] - case[idx-1][constants.attr_arrived_time]
            dwell_times[eqtyp_idx].append(dwell_time.total_seconds())
    
    # Calculate the average dwell time per EQTYP
    for i in range(len(unique_eqtyp)):
        average_dwell_time = np.mean([dwell_time for dwell_time in dwell_times[i]])
        average_dwell_time_per_eqtyp.append(average_dwell_time)
     
    # Create a dictionary mapping per EQTYP to its average dwell time
    return dict({eqtyp: dwell_time for eqtyp, dwell_time in zip(unique_eqtyp, average_dwell_time_per_eqtyp)})





