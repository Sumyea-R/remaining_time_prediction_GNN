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


def create_prefix_log(testing_log: EventLog, starting_prefix_size: int) -> EventLog:
    """
    Create a prefix log based on the testing log with a specified prefix size
    """
    prefix_log = EventLog([Trace(case[:starting_prefix_size], attributes={'concept:name': case.attributes['concept:name'], 
                                         'Src Eqt': case.attributes['Src Eqt'], 'Dest Eqt': case.attributes['Dest Eqt']}) 
                       for case in testing_log])
    return prefix_log

def denote_prefix_events(prefix_log: EventLog) -> None:
    """
    Denotes the events in the prefix log as non-predicted and stores the ground truth time.
    """
    for case in prefix_log:
        for event in case:
            event['is_predicted'] = False # Mark event as not predicted
            event['Ground Truth Time'] = event[constants.attr_arrived_time]

def append_prediction_to_case(prefix_case: Trace, predictions: [[]]) -> None:
    """
    Appends predictions to the prefix case by adding the predicted EID and arrival time.
    """
    for pred in predictions:

        eqtyp = ''.join(pred[1][0:2]) # Join the first two digits to form EQTYP
        prefix_case.append({constants.attr_equipment: pred[1], 'is_predicted': True, constants.attr_arrived_time: pred[2], 'EQTYP': eqtyp}) # Mark prediction     
            
def post_process(prefix_log: EventLog, prediction: [[str, str, datetime]]) -> EventLog:
    """
    Performs post-processing on the prefix log by adding predictions to the appropriate cases and events.
    """
    denote_prefix_events(prefix_log) # Denote an event if it is an input prefix event
    for case in prefix_log:
        case_prediction = [pred for pred in prediction if pred[0] == case.attributes['concept:name']]
        append_prediction_to_case(case, case_prediction) # Append prediction to the case
    return prefix_log

def add_ground_truth(testing_log: EventLog, prediction_log: EventLog) -> pd.DataFrame:
    """
    Adds ground truth information to the prediction log based on the testing log.
    """
    for case_test, case_predict in zip(testing_log, prediction_log):
        for event_test, event_predict in zip(case_test, case_predict):
            event_predict['Ground Truth'] = event_test[constants.attr_equipment] # Store ground truth EID
            event_predict['Ground Truth Time'] = event_test[constants.attr_arrived_time] # Store arrival time for computing the ground truth dwell time
            event_predict['further_prediction'] = False 
        if len(case_test) < len(case_predict):
            for event in case_predict[len(case_test):]:
                event['further_prediction'] = True
                
        if len(case_test) > len(case_predict):
            for i in range(len(case_predict), len(case_test)):
                ground_truth = case_test[i][constants.attr_equipment] # Ground truth EID
                ground_truth_time = case_test[i][constants.attr_arrived_time] # Arrival time for computing ground truth dwell time
                case_predict.append({constants.attr_case: case_test.attributes['concept:name'], 'Ground Truth': ground_truth, 
                                     'Ground Truth Time': ground_truth_time, 'is_predicted': False, 'further_prediction': False})
    return prediction_log





