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
    """
    take the salaries dataframe and extract the state from locations
    if the location is not in the US fill with NA
    return a dataframe with the column state
    """
    location_list = data['location'].values.tolist()
    fifty_state_list=['SD','PA','WV','MO','MT','NE','NV','NH','NJ','NM','NY','SC','NC','ND',
                'OH','OK','OR','TX','AL','VA','WI','RI','TN','WA','UT','VT','KY','IN',
                'MS','AK','AZ','AR','CA','CO','CT','DE','DC','KS','FL','GA','HI','ID',
                'IL','ME','WY','MI','LA','IA','MN','MA','MD']
    location_list = data['location'].values.tolist()
    state_list = [location.split(',')[1].strip() for location in location_list]
    new_state_list = []
    for value in state_list:
        if value not in fifty_state_list:
            print(value)
            new_state_list.append('NA')
        else:
            new_state_list.append(value)
    data_frame['state_cleaned'] = new_state_list
    return(data_frame)


# Takes a dataframe, filter column and specific cutoffs as input and ouputs a dictionary of lists where each list is the target column filtered.
def control_column(df, filter_column, return_column, cutoffs):
    dict_ = {}
    cutoffs = [0] + cutoffs + [max(df[filter_column])]
    
    for i in range(0, len(cutoffs)-1):
        dict_[i] = list(df[(df[filter_column].between(cutoffs[i], cutoffs[i+1], inclusive=True))][return_column])
    
    return dict_
