import pandas as pd
from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import EventLog, Trace, Event
import math
import csv
from bidict import bidict
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
import regex as re
import preprocess_graph_attributes
import constants

def encode_routing_features(eid_mapper: bidict, case: Trace, event: Event, network: nx.DiGraph, max_successors: int) -> [int]:
    """
    Encode successor equipments with lowest score in the routing network
    """
    routing_vector = []
    lowest_scored_successors = []
    equipment = event[constants.attr_equipment]
    src_loc = case.attributes['Src Loc']
    dest_loc = case.attributes['Dest Loc']
    for n in network.successors(equipment):
        neighbours = list(network[equipment][n].get('score'))
        neighbours = filter(lambda x: ((x[0] == src_loc) & (x[1] == dest_loc)), neighbours)
        neighbours = sorted(neighbours, key=lambda score: score[2])
        if neighbours:
            tup = (n, neighbours[0])
            lowest_scored_successors.append(tup)
    
    lowest_scored_successors = sorted(lowest_scored_successors, key=lambda x: x[1][2])
    successor = min(len(lowest_scored_successors), max_successors)
    
    for i in range(successor):
        encoded_eid = eid_mapper[lowest_scored_successors[i][0]]
        score_of_encoded_eid = lowest_scored_successors[i][1][2]
        routing_vector+= [encoded_eid, score_of_encoded_eid]
    
    routing_vector+= [-1]*(max_successors*2 - len(routing_vector))
    
    return routing_vector
