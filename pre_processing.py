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

def convert_timestamp(log: EventLog) -> EventLog:
    """
    Convert the timestamp from string to datetime
    """
    
    for case in log:
        # Format the 'Init Datetime' attribute as datetime
        #case.attributes['Init Datetime'] = datetime.strptime(case.attributes['Init Datetime'], '%d/%m/%y %H:%M')
        case.attributes['Init Datetime'] = datetime.strptime(case.attributes['Init Datetime'], '%Y-%m-%d %H:%M:%S')
        # Format the 'Complete Datetime' attribute as datetime
        #case.attributes['Complete Datetime'] = datetime.strptime(case.attributes['Complete Datetime'], '%d/%m/%y %H:%M')
        case.attributes['Complete Datetime'] = datetime.strptime(case.attributes['Complete Datetime'], '%Y-%m-%d %H:%M:%S')
        for event in case:
            # Convert the timestamp from string to datetime format
            #event[constants.attr_arrived_time] = datetime.strptime(event[constants.attr_arrived_time], '%d/%m/%y %H:%M')
            event[constants.attr_arrived_time] = datetime.strptime(event[constants.attr_arrived_time], '%Y-%m-%d %H:%M:%S')
    return log


def create_folds(log: EventLog, number_of_folds: int = 2) -> []:
    """
    Split the event log into multiple folds for cross-validation.
    """
    number_of_cases_per_folds = math.floor(len(log)/number_of_folds)
    folds = []
    curr_case_idx = 0

    # Create folds except the last one
    for fold_count in range(number_of_folds-1):
        folds.append(EventLog(log[curr_case_idx: curr_case_idx+number_of_cases_per_folds]))
        curr_case_idx += number_of_cases_per_folds

    # Create the last fold with remaining cases
    folds.append(EventLog(log[curr_case_idx:]))
    return folds

def adjust_arrived_time(log: EventLog) -> EventLog:
    """
    Shift Arrived time of events to indicate the departed time. 
    """
    for case in log:
        # Shift the arrival time of the second EID as the departure time of the first EID
        case[0][constants.attr_arrived_time] = case[1][constants.attr_arrived_time]
        for idx, event in enumerate(case[1:-1], 1):
            event[constants.attr_arrived_time] = case[idx + 1][constants.attr_arrived_time]
                    
        # Set the departure time of the last EID with the complete time of the case
        case[len(case) - 1][constants.attr_arrived_time] = case.attributes['Complete Datetime']
    return log

def get_training_log(folds: []) -> EventLog:
    
    # Combine the first two folds for training
     return EventLog([case for sub_log in folds[:2] for case in sub_log])
    
def get_testing_log(folds: []) -> EventLog:
    
    # Use the third fold for testing
    EventLog([case for sub_log in folds[2:] for case in sub_log])



