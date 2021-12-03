# DS_GA_1001_Capstone



## Predicting DS Salaries based on qualitative factors.
### By Giulio Duregon, Joby George, Howell Lu, Jonah Poczobutt, Alexandre Vives
### Due 12/22/201

# Goals of the project

The main goals of this project are to:

    1. Data Exploration and understanding
    2. Statistical analysis of the Dataset 
    3. Predict Data Science salaries
    
To further refine our understanding we will investigate the following questions:

## Specific Questions

    1) For when we have gender, do men make more than women when controlling for experience (Jonah)
    2) Highest total pay by location (Joby)
    3) Do people with high v low years at current co make more (Alex)
    4) Has the median annual salary increase for Data Science/Software Engineer increased more than annual inflation between 2017-2019
    5) How many years of experience is a Master's Degree worth (Howell)
    6) FAANG vs Non-FAANG (Tech) (Alex)
    7) Finance vs FAANG (Giulio)

    1) Predict salary from other factors
    2) Predict experience level from salary
    3) Predict company from salary info
    see how we perform on prediction when we drop some of these factors


## Background on Data

The dataset we are using for this project can be found on [kaggle](https://www.kaggle.com/jackogozaly/data-science-and-stem-salaries). The data was scraped from levels.fyi and had some cleaning done to it before being uploaded onto kaggle.

There are 62,000 rows and 29 columns, described below:

      1.
      2.
      3.
      4.
      5.
      6.
      7.
      8.
      9.
      10.
      11.
      12.
      13.
      14.
      15.
      16.
      17.
      18.
      19.
      20.
      21.
      22.
      23
      24.
      25.
      26.
      27.
      28.
      29.
      
      
      

## Motivating Factor for this Project

As Data Science graduate students at New York University's Center for Data Science, the team was naturally intrigued to learn what factors are most important with salaries of Data Science and STEM employees. 


for key in test_input2.keys():
    test = mannwhitneyu(test_input2[key][0], test_input2[key][1], alternative='two-sided') #Runs a Mann Whitney U-test
    if test.pvalue < 0.05:
        print('{} years of experience: We reject the Null Hypothesis (p-value = {})'.format(key, test.pvalue))
    else:
        print('{} years of experience: We fail to reject the Null Hypothesis (p-value = {})'.format(key, test.pvalue))
