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
import numpy as np
import pandas as pd
import math
import itertools 
import matplotlib.pyplot as plt
import seaborn as sns

#Takes in the number of features to plot and returns the x and y values for the join plot
def plotsize_check(features_length):
    if features_length == 2:
        result = (1, 2)
    elif features_length == 4:
        result = (2, 2)
    elif features_length == 5:
        result = (2, 3)
    else:
        if features_length % 4 == 0:
            result = (int(features_length/4), 4)
        elif features_length % 3 == 0:
            result = (int(features_length/3), 3)
        else:
            result = (int(np.ceil(features_length/4)), 4)

    return result[0], result[1]

#Takes the dataframe and a list with the categorical columns and plots them using seaborn's countplot
def plot_features(df, feature_list, is_categorical=True, compare_feature=None):
    rows, columns = plotsize_check(len(feature_list))
    fig, axes = plt.subplots(rows, columns, figsize=(7*columns, 8*rows))

    for i in range(0,rows):
        for j in range(0,columns):
            if i*columns + j == len(feature_list): break
            
            if is_categorical == True and compare_feature != None:
                sns.countplot(ax=axes[i, j], x=df[feature_list[i*columns+j]], hue = df[compare_feature], palette='mako')
                axes[i, j].set_title(feature_list[i*columns+j])
            elif is_categorical == True and compare_feature == None:
                sns.countplot(ax=axes[i, j], x=df[feature_list[i*columns+j]], palette='mako')
                axes[i, j].set_title(feature_list[i*columns+j])
            else: 
                sns.kdeplot(ax=axes[i, j], x=df[feature_list[i*columns+j]], shade=True)
                axes[i, j].set_title(feature_list[i*columns+j])

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
    data_frame = data_frame.loc[data_frame['state_clean']!='NA']
    data_frame.drop('state_clean', axis=1, inplace=True)
  
    return(data_frame)

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
    faang_list = ['facebook', 'apple', 'amazon', 'netflix', 'google']
    
    for i in range(0, len(cutoffs)-1):
        temp = df[(df[control_column1].between(cutoffs[i], cutoffs[i+1], inclusive='left')) & (df[control_column2] == 'Technology')]
        faang = temp[temp[filter_column].isin(faang_list)][return_column]
        nonfaang_tech = temp[~temp[filter_column].isin(faang_list)][return_column]
        
        dict_[str(cutoffs[i]) + '-' + str(cutoffs[i+1])] = [faang, nonfaang_tech]
    return dict_

def hypothesis_data3(df, bucket_list):
    
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

def hypothesis_data4(df,bucketlist):
    
    bucketlist = [0] + bucketlist + [100]
    filtered_df=df.dropna(subset = ['Education'])
    filtered_HS = filtered_df[filtered_df['Education'] == "Highschool"]
    filtered_HS = filtered_HS.sort_values('yearsofexperience')
    filtered_bachelor = filtered_df[filtered_df['Education'] == "Bachelor's Degree"]
    filtered_bachelor = filtered_bachelor.sort_values('yearsofexperience')
    filtered_master = filtered_df[filtered_df['Education'] == "Master's Degree"]
    filtered_master = filtered_master.sort_values('yearsofexperience')
    filtered_phd = filtered_df[filtered_df['Education'] == "PhD"]
    filtered_phd = filtered_phd.sort_values('yearsofexperience')
    
    hsarray = []
    collegearray = []
    masterarray = []
    phdarray = []
    i=0
    while i < len(bucketlist)-1:
            hsarray.append(list(filtered_HS['totalyearlycompensation'][(filtered_HS['yearsofexperience']>=bucketlist[i]) & (filtered_HS['yearsofexperience']<bucketlist[i+1])]))
            collegearray.append(list(filtered_bachelor['totalyearlycompensation'][(filtered_bachelor['yearsofexperience']>=bucketlist[i]) & (filtered_bachelor['yearsofexperience']<bucketlist[i+1])]))
            masterarray.append(list(filtered_master['totalyearlycompensation'][(filtered_master['yearsofexperience']>=bucketlist[i]) & (filtered_master['yearsofexperience']<bucketlist[i+1])]))
            phdarray.append(list(filtered_phd['totalyearlycompensation'][(filtered_phd['yearsofexperience']>=bucketlist[i]) & (filtered_phd['yearsofexperience']<bucketlist[i+1])]))
            i=i+1
         
    j = 0
    dict_ = {}    
    while j < len(hsarray):
        dict_[str(bucketlist[j]) + '-' + str(bucketlist[j+1])] = [hsarray[j],collegearray[j],masterarray[j],phdarray[j]]
        j += 1
        
    return dict_

def hypothesis_data5(data_frame, cutoff_array):
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
    try:
        region_keys.remove('NA')
    except:
        pass
    buckets = list(set(new_data_frame['levels_of_experience'].values.tolist()))
    final_keys =  list(itertools.product(region_keys, buckets))
    big_list = []
    for value in final_keys:
        big_list.append(new_data_frame.loc[(new_data_frame['region']== str(value[0])) & (new_data_frame['levels_of_experience']== value[1])]['totalyearlycompensation'].tolist())

    final_dict = dict(zip(final_keys, big_list))
    
    return(final_dict)


#Takes a dataframe, matches 161 company names for their industry, returns the altered dataframe
#Uses Sector Dict
def match_industry(df):
    for i in sector_dict:
        df.loc[df.company==str(i),'sector'] = sector_dict[i]
    return df

#Sector_dict used for match_industry
sector_dict = {'google': 'Technology', 'cisco': 'Technology', 'uber': 'Technology', 'capital one': 'Financials', 'linkedin': 'Technology', 'vmware': 'Technology', 'bloomberg': 'Technology', 'goldman sachs': 'Financials',
               'paypal': 'Technology', 'deloitte': 'Business Services', 'walmart labs': 'Technology', 'accenture': 'Business Services', 'expedia': 'Technology', 'sap': 'Technology',
               'dropbox': 'Technology', 'shopify': 'Technology', 'airbnb': 'Technology', 'atlassian': 'Technology', 'snap': 'Technology', 'yelp': 'Technology', 'yahoo': 'Technology', 'stripe': 'Technology',
               'indeed': 'Technology', 'yandex': 'Technology', 'bytedance': 'Technology', 'zillow': 'Technology', 'spotify': 'Technology', 't-mobile': 'Telecommunications', 'pinterest': 'Technology', 
               'nutanix': 'Technology', 'doordash': 'Technology', 'amd': 'Technology', 'ernst and young': 'Business Services', 'qualtrics': 'Technology', 'samsung': 'Technology', 'cruise': 'Transportation', 
               'twilio': 'Telecommunications', 'instacart': 'Technology', 'booking.com': 'Technology', 'box': 'Technology', 'pwc': 'Business Services', 'coinbase': 'Technology', 'disney': 'Media', 'citi': 'Financials', 
               'arm': 'Technology', 'flipkart': 'Retailing', 'epic systems': 'Technology', 'booz allen hamilton': 'Business Services', 'zalando': 'Technology', 'verizon': 'Telecommunications', 'slalom': 'Business Services',
               'tableau software': 'Technology', 'mckinsey': 'Business Services', 'hubspot': 'Technology', 'wework': 'Technology', 'robinhood': 'Financials', 'docusign': 'Technology', 'compass': 'Food & Drug Stores', 
               'slack': 'Technology', 'kpmg': 'Business Services', 'grab': 'Technology', 'citrix systems inc': 'Technology', 'two sigma': 'Financials', 'palantir': 'Technology', 'barclays': 'Financials', 'hpe': 'Technology', 
               'tata consultancy services': 'Business Services', 'cognizant': 'Business Services', 'amazon': 'Technology', 'rubrik': 'Technology', 'okta': 'Technology', 'zendesk': 'Technology', 'etsy': 'Technology', 
               'rally health': 'Health Care', 'capgemini': 'Business Services', 'github': 'Technology', 'infosys': 'Business Services', 'bank of america merrill lynch': 'Financials', 'roblox': 'Technology', 
               'mathworks': 'Technology', 'hulu': 'Technology', 'blizzard entertainment': 'Media', 'roku': 'Technology', 'grubhub': 'Technology', 'waymo': 'Technology', 'nokia': 'Technology', 'symantec': 'Technology', 
               'citadel': 'Financials', 'mediatek': 'MediaTek', 'fidelity investments': 'Financials', 'squarespace': 'Technology', 'datadog': 'Technology', 'ge digital': 'Technology', 'unity technologies': 'Technology', 
               'credit karma': 'Financials', 'adp': 'Financials', 'zoox': 'Motor Vehicles & Parts', 'akamai': 'Technology', 'publicis sapient': 'Business Services', 'cloudera': 'Technology', 'sofi': 'Financials', 
               'the home depot': 'Household Products', 'the new york times company': 'Media', 'optum': 'Health Care', 'audible': 'Technology', 'flexport': 'Industrials', 'nvidia': 'Technology', 'thoughtworks': 'Technology', 
               'wish': 'Technology',  'argo ai': 'Technology', 'bcg': 'Business Services', 'blue origin': 'Technology', 'shopee': 'Technology', 'aws': 'Technology', 'state farm': 'Financials', 'rivian': 'Technology', 
               'ericsson': 'Telecommunications', 'redfin': 'Technology', 'mozilla': 'Technology', 'marvell': 'Media', 'samsara': 'Technology', 'affirm': 'Financials', 'pivotal': 'Technology', 
               'deutsche bank': 'Financials', 'ey': 'Business Services', 'siemens': 'Technology', 'mongodb': 'Technology', 'bny mellon': 'Financials', 'klarna': 'Financials', 'exxonmobil': 'Energy', 'twitch': 'Technology', 
               'reddit': 'Technology', 'ubs': 'Financials', 'bain': 'Business Services', 'convoy': 'Technology', 'plaid': 'Financials', 'pure storage': 'Technology', 'vanguard': 'Financials', 'honeywell': 'Industrials', 
               'asml': 'ASML', 'riot games': 'Media', 'pandora': 'Technology', 'rakuten': 'Technology', 'delivery hero': 'ASML', 'factset': 'Financials', 'skyscanner': 'Technology', 'asurion': 'Technology', 
               'asana': 'Technology', 'carta': 'Technology', 'alibaba': 'Technology', 'informatica': 'Technology', 'databricks': 'Technology', 'surveymonkey': 'Technology', 'cox automotive': 'Motor Vehicles & Parts', 
               'ancestry': 'Technology', 'cloudflare': 'Technology', 'smartsheet': 'Technology', 'trend micro': 'Technology', 'gusto': 'Technology'}

def hypothesis_data6(df):
    
    # Significance tests
    from scipy.stats import mannwhitneyu
    #Grab salary data for companies that are in the Financials Sector
    financial_df = df[df['sector']=='Financials']
    
    #Grab salary data for companies that are in the Technology Sector
    technology_df = df[df['sector']=='Technology']
                          
    #Initialize two dicts to contain the arrays of data
    fin_dict = {}
    tech_dict = {}

    #Initialize list for year of experience buckets
    cutoffs = [0,5,10]
    
    #Loop through the financial salaries, and add to appropriate dictionary key
    for i in range(len(cutoffs)-1):
        fin_dict[str(cutoffs[i])+"-"+str(cutoffs[i+1])] = financial_df[(financial_df['yearsofexperience'] >= cutoffs[i]) & (financial_df['yearsofexperience'] < cutoffs[i+1])]['totalyearlycompensation']
    
    #Our loop doesn't grab the last band, so grab the last band outside the loop
    fin_dict[str(cutoffs[-1])+'+'] = financial_df[financial_df['yearsofexperience'] > cutoffs[-1]]['totalyearlycompensation']
    cutoffs = [0,5,10]
    
    #Loop through the tech salaries, and add to appropriate dictionary key
    for i in range(len(cutoffs)-1):
        tech_dict[str(cutoffs[i])+"-"+str(cutoffs[i+1])] = technology_df[(technology_df['yearsofexperience'] >= cutoffs[i]) & (technology_df['yearsofexperience'] < cutoffs[i+1])]['totalyearlycompensation']
    
    #Our loop doesn't grab the last band, so grab the last band outside the loop
    tech_dict[str(cutoffs[-1])+'+'] = technology_df[technology_df['yearsofexperience'] > cutoffs[-1]]['totalyearlycompensation']
    
    return tech_dict, fin_dict
 
def pca(data, stand=False, k=None, var=False):
    
    cols = list(data.iloc[:0])
    #Create a copy of data
    data_copy = data.copy()
    
    # Center Data at 0, Each Column must have mean 0
    data_copy = data - data.mean()

    # If we should standardize, then standardize the dataset
    if stand == True:
        data_copy = data_copy / data_copy.std()

    # Compute Covariance Matrix
    Cov_matrix = data_copy.T @ data_copy
    Cov_matrix = Cov_matrix / (len(data_copy)-1)
    # Calculate eigendecomp for Cov_Matrix
    vals, vectors = np.linalg.eigh(Cov_matrix)

    # Sort the eigenvalues descending order
    vals = vals[::-1]
    vectors = vectors[:, ::-1]

    # If we wanted the least dimensions for a certain amount of variance explained

    # See if Var was passed to function and its valid
    if var > 0 and var <= 1:
        tracker = 0
        eig_vals_of_interest, eig_vectors_of_interest = [], []
        total_var = vals.sum()
        for i in range(len(vals)):
            if tracker < var:
                tracker += vals[i] / total_var
                eig_vals_of_interest.append(vals[i])
        eig_vectors_of_interest = vectors[:, :len(eig_vals_of_interest)]

    # Check if we just wanted k dimensions
    elif k > 0 and k <= len(cols):

        # If we wanted k dimensions, set the appropriate amount
        eig_vals_of_interest = vals[:k]
        eig_vectors_of_interest = vectors[:, :k]
    else:
        return vals, vectors

    # Make sure data types are compatible
    eig_vectors_of_interest = np.array(eig_vectors_of_interest)

    # Compute the data projected onto the pcas
    new_data = data_copy@eig_vectors_of_interest

    #Returns the data projected onto the pcas, the eigenvalues, and the eigenvectors (PCs) of the covariance matrix
    return new_data, eig_vals_of_interest, eig_vectors_of_interest


def ForwardSelection(df_features, target):
    from pingouin import linear_regression
    
    remaining_features = list(df_features.columns)
    chosen_features = list()

    while len(remaining_features) > 0:
        selected_feature = None
        best_SSE = 0

        for feature in remaining_features:
            try_model = pd.concat([df_features[chosen_features], df_features[feature]], axis=1)

            linreg = linear_regression(try_model, target)
            y_train_pred = pd.Series(list((try_model @ linreg['coef'][1:].array + linreg['coef'][0]).values))

            feature_SSE = np.sum((target.mean()-y_train_pred)**2)
            feature_pval = float(linreg['pval'][-1:])

            if feature_pval < 0.05 and best_SSE < feature_SSE:
                best_SSE = feature_SSE
                best_pvalue = feature_pval
                selected_feature = feature
                
        print('Chosen feature: {} (with SSE: {} and p-value: {})'.format(selected_feature, int(best_SSE), round(feature_pval,5)))
        
        if selected_feature == None: break
            
        chosen_features.append(selected_feature)   
        remaining_features.remove(selected_feature)
        
    return chosen_features

def faang(df):
    faang_list = ['facebook', 'apple', 'amazon', 'netflix', 'google']
    
    df['faang'] = np.where(~df['company'].isin(faang_list), 'non_faang', 'faang')
    return df

def extract_year(df):
    #have to call this function before we drop timestamp
    df['timestamp'] = pd.to_datetime(df.timestamp, format= "%m/%d/%Y %H:%M:%S")
    df['year'] = df['timestamp'].dt.year
    return(df)

def extract_tag(df):
    tag_list = df['tag'].value_counts()[0:10].index.tolist()
    tag_values = df['tag'].value_counts()[0:10].index.tolist()
    other_tag_keys = df['tag'].value_counts()[10:].index.tolist()
    other_tag = ['other']*len(other_tag_keys)
    tag_list.extend(other_tag_keys)
    tag_values.extend(other_tag)
    tag_dict = dict(zip(tag_list, tag_values))

    for i in tag_dict:
        df.loc[df.tag==str(i),'tag'] = tag_dict[i]
    return(df)

