import networkx as nx
import regex as re

def preprocess_edge_attributes(G=nx.DiGraph):
	for edge in G.edges():
		if G[edge[0]][edge[1]]['score'] != '{(-1, -1, -1)}':
			data = G[edge[0]][edge[1]]['score']
			data = data.replace('{','')
			data = data.replace('}','')

			str_tuples = re.findall(r'\(.*?\)', data)
			score_set = set()
			for item in str_tuples:
				values_str = re.findall(r'[0-9A-Z]+', item)
				converted_tuple = tuple([values_str[0], values_str[1], int(values_str[2])])
				score_set.add(converted_tuple)

			G[edge[0]][edge[1]].update(score=score_set)

		else:
			G[edge[0]][edge[1]].update(score={(-1,-1,-1)})
			

def preprocess_graph_attributes(path_to_nw_file:str):
	'''
    Input: NetworkX gml file 
    Returns: desrialised DiGraph object
    '''
	logistic_nw = nx.read_gml(path_to_nw_file)
	preprocess_edge_attributes(logistic_nw)
	return logistic_nw