"""
Helper Functions.py


A place to upload helper functions that will be used in the 
DS GA 1001 Capstone Project

Changelog:
    v0: Joby initialized the file
    v1: Jonah person inputted percentile breaker and control for experience functions
    v2: Joby created extract state function
    v3: Alex and Jonah created awesome function to get us dictionary's of that contain sub-groups we want to perform hypothesis on testing on
    v4. Joby created function that outputs a dictionary that outputs regional salaries, controlling for experience
    ...
"""

import math
import itertools 

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

def hypothesis_data_3(data_frame, cutoff_array):
    """
    takes in a dataframe and creates a region column for the data, if it does not exist continue,
    input an array to look at experience cutoff groups and then return a dictionary that 
    has a key of the region and the experience group, with associated value of an array
    of the total yearly compensation for all employees in that region and experience group 
    """
    if 'region' in data_frame:
        pass
    else:
        new_data_frame = extract_state(data_frame).sort_values('yearsofexperience', ascending = True)
        
    cutoffs = control_for_experience(data_frame, cutoff_array)
    
    i = len(cutoff_array)
    
    while i > 0:
        if i == len(cutoff_array):
            levels_of_experience = ['0'] * cutoffs[len(cutoff_array) - i]
            i-=1
        else:
            new_list =  [len(cutoff_array) - i] * (cutoffs[len(cutoff_array)-i]-cutoffs[len(cutoff_array)-i-1])
            levels_of_experience += new_list
            i-=1
    if i == 0:
        new_list =  [len(cutoff_array)] * (len(data_frame) - cutoffs[-1])
        levels_of_experience += new_list
        
    new_data_frame = data_frame.sort_values('yearsofexperience', ascending = True)
    new_data_frame['levels_of_experience'] = levels_of_experience    
    region_keys = list(set(new_data_frame['region'].values.tolist()))
    region_keys.remove('NA')
    buckets = list(set(new_data_frame['levels_of_experience'].values.tolist()))
    final_keys =  list(itertools.product(region_keys, buckets))
    big_list = []
    for value in final_keys:
        big_list.append(new_data_frame.loc[(new_data_frame['region']== str(value[0])) & (new_data_frame['levels_of_experience']== str(value[1]))]['totalyearlycompensation'].tolist())

    final_dict = dict(zip(final_keys, big_list))
    
    return(final_dict)


# Takes a dataframe, filter column and specific cutoffs as input and ouputs a dictionary of lists where each list is the target column filtered.
def hypothesis_data1(df, control_column, filter_column, return_column, cutoffs):
    dict_ = {}
    cutoffs = [0] + cutoffs + [max(df[control_column])]
    
    for i in range(0, len(cutoffs)-1):
        temp = df[(df[control_column].between(cutoffs[i], cutoffs[i+1], inclusive='left'))]
        low = temp[temp[filter_column] <= temp[filter_column].median()][return_column]
        high = temp[temp[filter_column] > temp[filter_column].median()][return_column]
        
        dict_[str(cutoffs[i]) + '-' + str(cutoffs[i+1])] = [low, high]
    
    return dict_


# Takes a dataframe, filter columns and specific cutoffs as input and ouputs a dictionary of lists where each list is the target column filtered.
def hypothesis_data2(df, control_column1, control_column2, filter_column, return_column, cutoffs):
    dict_ = {}
    cutoffs = [0] + cutoffs + [max(df[control_column1])]
    faang_list = ['Facebook', 'Apple', 'Amazon', 'Netflix', 'Google']
    
    for i in range(0, len(cutoffs)-1):
        temp = df[(df[control_column1].between(cutoffs[i], cutoffs[i+1], inclusive='left')) & (df[control_column2] == 'Technology')]
        faang = temp[temp[filter_column].isin(faang_list)][return_column]
        nonfaang_tech = temp[~temp[filter_column].isin(faang_list)][return_column]
        
        dict_[str(cutoffs[i]) + '-' + str(cutoffs[i+1])] = [faang, nonfaang_tech]
    
    return dict_
#Takes a dataframe, matches 161 company names for their industry, returns the altered dataframe
#Uses Sector Dict
def match_industry(df):
    for i in sector_dict:
        df.loc[df.company==str(i),'sector'] = sector_dict[i]
    return df

#Sector_dict used for match_industry
sector_dict = {'Google':'Technology',
'Cisco':'Technology',
'Uber':'Technology',
'Capital One':'Financials',
'LinkedIn':'Technology',
'VMware':'Technology',
'Bloomberg':'Technology',
'Goldman Sachs':'Financials',
'PayPal':'Technology',
'Deloitte':'Business Services',
'Walmart Labs':'Technology',
'Accenture':'Business Services',
'Expedia':'Technology',
'SAP':'Technology',
'Dropbox':'Technology',
'Shopify':'Technology',
'Airbnb':'Technology',
'Atlassian':'Technology',
'Snap':'Technology',
'Yelp':'Technology',
'Yahoo':'Technology',
'Stripe':'Technology',
'Indeed':'Technology',
'Yandex':'Technology',
'ByteDance':'Technology',
'Zillow':'Technology',
'Spotify':'Technology',
'T-Mobile':'Telecommunications',
'Pinterest':'Technology',
'Nutanix':'Technology',
'DoorDash':'Technology',
'AMD':'Technology',
'Ernst and Young':'Business Services',
'Qualtrics':'Technology',
'Samsung':'Technology',
'Cruise':'Transportation',
'Twilio':'Telecommunications',
'Instacart':'Technology',
'Booking.com':'Technology',
'Box':'Technology',
'PwC':'Business Services',
'Coinbase':'Technology',
'Disney':'Media',
'Citi':'Financials',
'Arm':'Technology',
'Flipkart':'Flipkart',
'Epic Systems':'Technology',
'Booz Allen Hamilton':'Business Services',
'Zalando':'Technology',
'Verizon':'Telecommunications',
'Slalom':'Business Services',
'Tableau Software':'Technology',
'McKinsey':'Business Services',
'HubSpot':'Technology',
'WeWork':'Technology',
'Robinhood':'Financials',
'DocuSign':'Technology',
'Compass':'Food & Drug Stores',
'Slack':'Technology',
'KPMG':'Business Services',
'Grab':'Technology',
'Citrix Systems Inc':'Technology',
'Two Sigma':'Financials',
'Palantir':'Technology',
'Barclays':'Financials',
'HPE':'Technology',
'Tata Consultancy Services':'Business Services',
'Cognizant':'Business Services',
'amazon':'Technology',
'Rubrik':'Technology',
'Okta':'Technology',
'Zendesk':'Technology',
'Etsy':'Technology',
'Rally Health':'Health Care',
'Capgemini':'Business Services',
'GitHub':'Technology',
'Infosys':'Business Services',
'Bank of America Merrill Lynch':'Financials',
'Roblox':'Technology',
'MathWorks':'Technology',
'Hulu':'Technology',
'Blizzard Entertainment':'Media',
'Roku':'Technology',
'Grubhub':'Technology',
'Waymo':'Technology',
'Nokia':'Nokia',
'Symantec':'Technology',
'Citadel':'Financials',
'MediaTek':'MediaTek',
'Fidelity Investments':'Financials',
'SquareSpace':'Technology',
'Datadog':'Technology',
'GE Digital':'Technology',
'Unity Technologies':'Technology',
'Credit Karma':'Financials',
'ADP':'Financials',
'Zoox':'Motor Vehicles & Parts',
'Akamai':'Technology',
'Publicis Sapient':'Business Services',
'Cloudera':'Technology',
'SoFi':'Financials',
'The Home Depot':'Household Products',
'The New York Times Company':'Media',
'Optum':'Health Care',
'Audible':'Technology',
'Flexport':'Industrials',
'NVIDIA':'Technology',
'ThoughtWorks':'Technology',
'Linkedin':'Technology',
'Wish':'Technology',
'Argo AI':'Technology',
'BCG':'Business Services',
'Blue Origin':'Technology',
'Shopee':'Technology',
'AWS':'Technology',
'State Farm':'Financials',
'Rivian':'Technology',
'Ericsson':'Telecommunications',
'Redfin':'Technology',
'Mozilla':'Technology',
'Marvell':'Media',
'Samsara':'Technology',
'Affirm':'Financials',
'Pivotal':'Technology',
'Deutsche Bank':'Financials',
'EY':'Business Services',
'Siemens':'Technology',
'MongoDB':'Technology',
'BNY Mellon':'Financials',
'Klarna':'Financials',
'ExxonMobil':'Energy',
'Twitch':'Technology',
'Reddit':'Technology',
'UBS':'Financials',
'Bain':'Business Services',
'Convoy':'Technology',
'Plaid':'Financials',
'Pure Storage':'Technology',
'Vanguard':'Financials',
'Honeywell':'Industrials',
'ASML':'ASML',
'Riot Games':'Media',
'Pandora':'Technology',
'Rakuten':'Technology',
'Delivery Hero':'ASML',
'FactSet':'Financials',
'Skyscanner':'Technology',
'Asurion':'Technology',
'Asana':'Technology',
'Carta':'Technology',
'Alibaba':'Technology',
'Informatica':'Technology',
'Databricks':'Technology',
'SurveyMonkey':'Technology',
'Cox Automotive':'Motor Vehicles & Parts',
'Ancestry':'Technology',
'Cloudflare':'Technology',
'google':'Technology',
'Smartsheet':'Technology',
'Trend Micro':'Technology',
'Gusto':'Technology'}
