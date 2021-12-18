# DS_GA_1001_Capstone



## Predicting DS Salaries based on qualitative factors.
### By Giulio Duregon, Joby George, Howell Lu, Jonah Poczobutt, Alexandre Vives
### Due 12/22/201

# Write-up Location

For a full write-up of the project, go to the following [google docs](https://docs.google.com/document/d/1cNr9mMmF3b0EuZzc6ek0Cd8W22jxeAhYq-UoMxkGgTk/edit?usp=sharing)

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
    7) Finance vs Tech (Giulio)

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


    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RepeatedKFold
    from sklearn.linear_model import LassoCV

    x_train, x_test, y_train, y_test = train_test_split(
    onehot_df_salary, target_salary, test_size=0.1, random_state=42)

    cva = RepeatedKFold(n_splits=10,n_repeats = 3, random_state=42)



    model = LassoCV(alphas=np.arange(0.001,1,.0001), cv=cva,n_jobs=-1)
    model.fit(x_train,y_train)
    print(model.alpha_)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=101)
    kf.get_n_splits(onehot_df_salary)


    train_error, test_error, R_squared = [], [], []

    for train_index, test_index in kf.split(onehot_df_salary):
        X_train, X_test = onehot_df_salary.iloc[train_index], onehot_df_salary.iloc[test_index]
        y_train, y_test = target_salary.iloc[train_index], target_salary.iloc[test_index]


        #linreg = sm.OLS(y_train,sm.add_constant(X_train)).fit()

        lasso = linear_model.Lasso(alpha=.001)
        lasso.fit(X_train, y_train)

        train_ypred = lasso.predict(X_train)

        test_ypred = lasso.predict(X_test)

        train_error.append(mean_squared_error(train_ypred,y_train))

        test_error.append(mean_squared_error(test_ypred,y_test))

        R_squared.append(lasso.score(X_test, y_test))

        #y_train_pred = linreg.predict(sm.add_constant(X_train[column_subset]))

        #y_test_pred = linreg.predict(sm.add_constant(X_test[column_subset]))


    print('Training MSE for this Lasso Regression model: {}'.format(round(np.mean(train_error), 3)))
    print('Testing MSE for this Lasso Regression model: {}\n'.format(round(np.mean(test_error), 3)))
    print('R^2: {}\n'.format(round(np.mean(R_squared), 3)))

    #print(linreg.summary())

    lasso_df = pd.DataFrame(lasso.coef_, onehot_df_salary.columns, columns=['regression_coef'])
    
    #in new cells, you can keep them commented out
    display(lasso_df.sort_values(by = 'regression_coef')[0:30])
    display(lasso_df.sort_values(by = 'regression_coef')[30:])
