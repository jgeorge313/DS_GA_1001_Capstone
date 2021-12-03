"""
Helper Functions.py


A place to upload helper functions that will be used in the 
DS GA 1001 Capstone Project

Changelog:
    v0: Joby initialized the file
    v1: Jonah person inputted percentile breaker and control for experience functions
    v2: Joby created extract state function
    ...
"""

import math
def percentile_breaker(df, percentile_break, col_name):
    #breaks sorted dataset on a certain percentile...this will be useful for testing salaries at small v large companies, ask Jonah for more info
    #returns dataframe split at proper percentile
    
    percentile_break = percentile_break/100
    breakpoint = math.ceil(percentile_break*df.shape[0])
    df= df.sort_values(col_name)
    group_1 = df.iloc[:breakpoint,:]
    group_2 = df.iloc[breakpoint:,:]
    
    return(group_1, group_2) 

def control_for_experience(df, buckets):
    #returns index breakpoints
    #need to pass cutoffs to sorted df for this to work, folks ***important***
    i=0
    j=0
    sorted_df = df.sort_values('yearsofexperience', ascending = True)
    vals = list(sorted_df['yearsofexperience'])
    cutoffs = []
    
    while i<len(buckets):
        
        while vals[j] <= buckets[i]:
            
            j+=1
        
        cutoffs.append(j)
        
        i+=1
    
    return (cutoffs)

def gender_control_experience_buckets(df, bucket_list):
    
    #bucket list is an array of buckets to group respondents by
    #returns dictionary with experience bucket top value key and 2 arrays of salaries [female, male] as value
    #relies on Jonah's original control for experience
    
    final_dict = dict()
    filtered_df=df.dropna(subset = ['gender'])
    filtered_df.sort_values('yearsofexperience')
    filtered_male = filtered_df[filtered_df['gender'] == 'Male']
    filtered_female = filtered_df[filtered_df['gender'] == 'Female']
    filtered_male = filtered_male.sort_values('yearsofexperience')
    filtered_female = filtered_female.sort_values('yearsofexperience')
    
    male_cutoffs = control_for_experience(filtered_male, bucket_list)
    female_cutoffs = control_for_experience(filtered_female, bucket_list)
    
    i=0
    while i < len(bucket_list):
        
        male_array = list(filtered_male['totalyearlycompensation'])[:male_cutoffs[i]]
        female_array = list(filtered_female['totalyearlycompensation'])[:female_cutoffs[i]]
        final_dict[bucket_list[i]]= [female_array, male_array]
        i+=1
        
    return (final_dict)
        
def extract_state(data_frame):
def extract_state(data_frame):
    """
    take the salaries dataframe and extract the state from locations
    if the location is not in the US fill with NA
    return a dataframe with the column state
    """
    fifty_state_list = ['AK','AL','AR','AZ','CA','CO','CT',
                  'DC','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS',
                  'MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA',
                  'RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']

    region_list = ['West','South','South','West','West','West','Northeast','South','South',
                   'South','South','West','Midwest','West','Midwest','Midwest',
                   'Midwest','South','South','Northeast','South','Northeast',
                   'Midwest','Midwest','Midwest','South','West','South','Midwest',
                   'Midwest','Northeast','Northeast','West','West','Northeast',
                   'Midwest','South','West','Northeast','Northeast','South',
                   'Midwest','South','South','West','South','Northeast',
                   'West','Midwest','South','West']
    
    geo_dict = dict(zip(fifty_state_list, region_list))

    location_list = data_frame['location'].values.tolist()
    state_list = [location.split(',')[1].strip() for location in location_list]
    new_state_list, new_region_list  = [], []
    for value in state_list:
        if value not in fifty_state_list:
            new_state_list.append('NA')
            new_region_list.append('NA')
        else:
            new_state_list.append(value)
            new_region_list.append(geo_dict[value])
    data_frame['state_clean'] = new_state_list
    data_frame['region'] = new_region_list

    return(data_frame)


# Takes a dataframe, filter column and specific cutoffs as input and ouputs a dictionary of lists where each list is the target column filtered.
def control_column(df, filter_column, return_column, cutoffs):
    dict_ = {}
    cutoffs = [0] + cutoffs + [max(df[filter_column])]
    
    for i in range(0, len(cutoffs)-1):
        dict_[i] = list(df[(df[filter_column].between(cutoffs[i], cutoffs[i+1], inclusive=True))][return_column])
    
    return dict_
