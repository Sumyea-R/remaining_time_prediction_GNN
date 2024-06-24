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
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import mappers
import constants

def calculate_dwell_times(prediction_log: EventLog, eqtyp_mapper: bidict, starting_prefix_size: int) -> []:
    """
    Calculate the predicted and true dwell times for each EQTYP.
    """
    pred_dwell_times = []
    true_dwell_times = []
    
    for i in range(len(eqtyp_mapper)):
        pred_dwell_times.append([])
        true_dwell_times.append([])
    
    for case in prediction_log:
        for idx, event in enumerate(case[starting_prefix_size:], starting_prefix_size):
            eqtyp = event['EQTYP']
            eqtyp_idx = eqtyp_mapper[eqtyp]
            if not event['further_prediction']:
                if event['is_predicted']:
                    # Calculate the predicted dwell time by subtracting the current event's
                    # arrival time from the previous event's arrival time
                    pred_dwell_time = event[constants.attr_arrived_time].timestamp() - case[idx - 1][constants.attr_arrived_time].timestamp()
                    pred_dwell_times[eqtyp_idx].append(pred_dwell_time)
                else:
                    pred_dwell_times[eqtyp_idx].append(0)
                        
            # Calculate the true dwell time by subtracting the current event's ground truth
            # time from the previous event's ground truth time
                true_dwell_time = event['Ground Truth Time'].timestamp() - case[idx - 1]['Ground Truth Time'].timestamp()
                true_dwell_times[eqtyp_idx].append(true_dwell_time)
    return [true_dwell_times, pred_dwell_times]


def calculate_mae_mape_per_eqtyp(dwell_times: [], dwell_times_mape: [], eqtyp_mapper: bidict) -> dict:
    """
    Calculate the Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE)
    for each equipment type based on the given dwell times.
    """
    mae_per_eqtyp = []
    mape_per_eqtyp = []
                    
    for i in range(len(dwell_times[0])):
        if dwell_times[1][i]:
            # Calculate MAE and MAPE using the true and predicted dwell times
            mae = metrics.mean_absolute_error(dwell_times[1][i], dwell_times[0][i])
            mae_per_eqtyp.append(mae)
            mape = np.mean(np.abs((np.array(dwell_times_mape[1][i]) - np.array(dwell_times_mape[0][i]))) / np.array(dwell_times_mape[0][i]))*100
            mape_per_eqtyp.append(mape)
        else:
            mae_per_eqtyp.append(0)
            mape_per_eqtyp.append(0)
    
    # Create dictionaries to map equipment types to their respective MAE and MAPE values        
    mae_mapper = dict({eqtyp : mae for eqtyp, mae in zip(eqtyp_mapper.keys(), mae_per_eqtyp)})
    mape_mapper = dict({eqtyp : mape for eqtyp, mape in zip(eqtyp_mapper.keys(), mape_per_eqtyp)})
    
    return dict({'MAE': mae_mapper, 'MAPE': mape_mapper})

def remove_null_dwell_times(dwell_times: [], eqtyp_mapper) -> []:
    """
    Remove the ground truth dwell times that are zero for better MAPE calculation.
    """
    null_dwell_times = np.zeros(len(dwell_times[0]))
    edited_true_dwell_times = []
    edited_pred_dwell_times = []
    
    for i in range(len(dwell_times[0])):
        if dwell_times[1][i]:
            true_dwell_time = []
            pred_dwell_time = []
            #remove true dwell times that are zero
            for j in range(len(dwell_times[1][i])):
                if dwell_times[0][i][j] != 0:
                    true_dwell_time.append(dwell_times[0][i][j])
                    pred_dwell_time.append(dwell_times[1][i][j])
                else:
                    #count how many dwell times have been removed
                    null_dwell_times[i]+= 1
            edited_true_dwell_times.append(true_dwell_time)
            edited_pred_dwell_times.append(pred_dwell_time)
    
    null_dwell_time_mapper = dict({eqtyp: null_dwell_time for eqtyp, null_dwell_time in zip(eqtyp_mapper.keys(), null_dwell_times)})
    print("Number of null dwell times removed per eqtyp")
    print(null_dwell_time_mapper)
    return [edited_true_dwell_times, edited_pred_dwell_times]
            
    
    
def calculate_mae_mape(prediction_log: EventLog, starting_prefix_size: int) -> dict:
    """
    Calculate the Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE)
    for each equipment type and overall based on the given dwell times.
    """
    
    # Filter the log to retain only the predicted events
    filtered_log = pm4py.filter_event_attribute_values(prediction_log, "is_predicted", [True], 
                                                       level="event", retain=True)
    # Build the equipment type mapper
    eqtyp_mapper = mappers.build_case_attr_mapper(filtered_log, 'EQTYP')

    # Calculate the dwell times per equipment type
    dwell_times_per_eqtyp = calculate_dwell_times(prediction_log, eqtyp_mapper, starting_prefix_size)
    dwell_times_per_eqtyp_mape = remove_null_dwell_times(dwell_times_per_eqtyp, eqtyp_mapper)

    # Calculate the MAE and MAPE per equipment type
    error_mappers = calculate_mae_mape_per_eqtyp(dwell_times_per_eqtyp, dwell_times_per_eqtyp_mape, eqtyp_mapper)

    # Flatten the true and predicted dwell times lists   
    true_dwell_times = [j for i in dwell_times_per_eqtyp[0] for j in i]
    pred_dwell_times = [j for i in dwell_times_per_eqtyp[1] for j in i]
    true_dwell_times_mape = [j for i in dwell_times_per_eqtyp_mape[0] for j in i]
    pred_dwell_times_mape = [j for i in dwell_times_per_eqtyp_mape[1] for j in i]
    
    # Calculate the overall MAE and MAPE
    mae = metrics.mean_absolute_error(np.array(true_dwell_times), np.array(pred_dwell_times))
    mape = np.mean(np.abs((np.array(true_dwell_times_mape) - np.array(pred_dwell_times_mape))) / np.array(true_dwell_times_mape))*100
    
    # Update the error mappers with the overall MAE and MAPE
    error_mappers['MAE'].update({'Overall MAE': mae})
    error_mappers['MAPE'].update({'Overall MAPE': mape})
    
    return error_mappers

def calculate_accuracy(result_df: pd.DataFrame, number_of_cases: int, starting_prefix_size: int) -> float:
    """
    Calculate the accuracy of equipment prediction based on the provided ground truth values.
    """

    cumulative_prefix_length = len(result_df) - starting_prefix_size*number_of_cases
    accurately_predicted_equipments = 0
    cumulative_mae = 0
    for idx, row in result_df.iterrows():
        if(row['is_predicted']): # Check if the row represents a predicted EID
            predicted_equipment = row[constants.attr_equipment] # Get the predicted EID
            ground_truth = row['Ground Truth'] # Get the ground truth
            if((predicted_equipment and ground_truth is not None) and (predicted_equipment == ground_truth)):
                accurately_predicted_equipments +=1 # Increment the counter for predictions
    accuracy = (accurately_predicted_equipments / cumulative_prefix_length) * 100 
    print('accuracy of equipment prediction:', accuracy)
    return accuracy





