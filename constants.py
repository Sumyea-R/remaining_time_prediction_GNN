# ### Required attributes from the dataset for prediction 

attr_caseid = 'Order OID'
attr_case = 'case:concept:name' #Order OID in xes format
attr_equipment = 'Equipment ID'
attr_arrived_time = 'Arrived Timestamp'
attr_init_time = 'case:Init Datetime'
attr_complete_time = 'case:Complete Datetime'
attr_fridge = 'is_fridge'
src_dest_attributes = ['case:Src Loc', 'case:Dest Loc']
required_attributes = [attr_case, attr_equipment,attr_arrived_time, attr_init_time, attr_complete_time]


event_attribute_features = ['EQTYP', 'Zone Name', 'Floor']
case_attribute_features = ['Src Eqt', 'Dest Eqt']
case_attribute = ['case:Src Eqt', 'case:Dest Eqt'] #case attribute in xes format
models_path = './Models_Routing/'
df_path = './Src_Dest_Loc_dfs/'
sampled_df_path = './Sampled_dfs/'

