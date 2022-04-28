# standards and customs
import pandas as pd
import numpy as np
import describe as tatl
import wrangle as get
import model as mod
from collections import Counter
# plotting and stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
# feature selection, splitting, scaling
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# import evaluation metrics and modeling methods
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def wrangle_df2(target):
    ''' 
    '''
    df, train, validate, test = wrangle_df()
    # first drop the columns I won't be using and that are unscalable. 
    train = train.drop(columns=(['country', 'year', 'year_int']))
    validate = validate.drop(columns=(['country', 'year', 'year_int']))
    test = test.drop(columns=(['country', 'year', 'year_int']))
    
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    # Change series into data frame for y 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test

def wrangle_df():
    df = prepare_data(acquire_data())
    df = create_time_df(df)
    train, validate, test = split_time_with_val(df)
    return df, train, validate, test

def clean_df():
    df = prepare_data(acquire_data())
    return df

def acquire_data():
    df = pd.read_csv('combined.csv')
    return df

def prepare_data(df):
    # Still considering playing with this as date-time.
    df['year'] = df['year'].fillna(0).astype('int')
    df = df.rename(columns={'Both sexes': 'both_sexes', 'Female': 'female', 'Male': 'male'})
    df['both_sexes'] = pd.DataFrame(df.both_sexes.str[0:4])
    df['female'] = pd.DataFrame(df.female.str[0:4])
    df['male'] = pd.DataFrame(df.male.str[0:4])
    # These two are unsalvageable 
    df = df.drop(['cor_exp', 'statcap'], axis=1)
    # Many columns that are of interest to be are nulls for these three countries
    # There's no way to properly infer the values 
    df = df[df['country'].apply(lambda val: all(val != s for s in ['Belarus', 'Bhutan', 'Guyana']))]
    # The mean has the closest alignment with the presumed reality in Ukraine and Kazakhstan
    # as well as the shape and distribution before imputation
    df = df.reset_index(drop=True)    
    df['avh'] = df.avh.fillna(df.avh.mean())
    # these columns offer no value to this study; will be kept in data dictionary
    # notes.md will explain why.
    df = df.drop(['i_cig', 'i_xm','i_xr', 'i_outlier', 'i_irr'], axis=1)
    df = df.astype({'both_sexes': 'float64', 'female': 'float64', 'male': 'float64'})
    # the following was added during exploration, due to heatmap observations correlating to both_sexes.
    # none were over a correlation of .37
    elimination = ['csh_m', 'xr', 'rkna', 'avh', 'hc', 'pop', 'csh_x', 'emp', 'pl_i', 'pl_n',\
                 'pl_m', 'male', 'female', 'countrycode', 'currency_unit']
    # these were removed because they were all nearly perfectly correlated with CGDPe; the redundancy may cause 
    # trouble for a model, and sicne they are essentially the same, there is no need to retain them. 
    demolition = ['rgdpe', 'rgdpo', 'ccon', 'cda',  'cgdpo', 'cn', 'rgdpna', 'rdana', 'ck', 'rconna', 'rnna']
    # add the lists to one another
    dropit = demolition + elimination
    df = df.drop(columns=dropit, axis=1)
    
    return df

def create_time_df(df):
    '''
    Takes in the suicide dataframe from prepare, creates a copy
    and then converts the year column from int to date-time format,
    with the unique year with hours timestamp set as the index.
    '''
    df2 = df.copy()
    
    # A particular approach to creating date time while avoiding pandas default to 1970
    df2['year'] = pd.to_datetime(df2['year'], format = "%Y").dt.strftime('%Y')
    # before adding additional elements I will create a column to be used as the index
    df2['date_index'] = df2['year']
    # guarantee it's a datetime 64 format
    df2['year'] = df2['year'].apply(pd.to_datetime)
    # create another as an int
    df2['year_int'] = df2['year'].astype(str).str[0:4]
    df2['year_int'] = df2['year_int'].astype(int)
    # now play with date time so I can add hours and make it into a unique index
    df2.date_index = pd.to_datetime(df2.date_index, infer_datetime_format = True)
    # add an hour to each recurrent year. 
    df2['date_index'] = df2['date_index'] + pd.to_timedelta(df2.groupby('date_index').cumcount(), unit='h')
    # set the index and sort it by date
    df2 = df2.set_index('date_index').sort_index()
    
    return df2

def split_time(time_df):
    '''
    Feed the time df and get a 70/30 train test dataframe by slicing the index.
    '''
    time_train = time_df[:'2013']
    time_test = time_df['2014':]
    return time_train, time_test

def split_time_with_val(df):
    '''
   Using methods gleaned from Codeup's Time Series Analysis lessons,
   splits the Superstore DF into Train Validate and Split; .5/.3/.2 respectively.
   Subsequently returns each.
    '''
    #print('Dataframe Input received: Splitting Data .5/.3/.2.')
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train_time = df[: train_size]
    validate_time = df[train_size : validate_end_index]
    test_time = df[validate_end_index : ]
    #print(f'Train: {train_time.shape}, Validate {validate_time.shape}, and Test {test_time.shape} are ready.\
    #\n Proceed with EDA.')
    return train_time, validate_time, test_time

def stats_visualized(train):
        # set alpha
    α = .05
    # conduct pearsonr test on independent and dependent variables.
    r, p = stats.pearsonr(train.ctfp, train.both_sexes)
    
    # plotting the variables used in the aforementioned test
    train.plot.scatter('ctfp', 'both_sexes')
    # Setting the title based on data dictionary
    plt.title('Total Factor Productivity (TFP) at Current PPP: Correlation to Suicide Rate')
    # displaying r at X and y positions, to the thousandths-place.
    plt.text(1, 40, f'r = {r:.3f}')
    # Boolean test on whether p is less than alpha. if true, the correlation is significant.
    print(f' P is less than alpha: {p < α}')
    # show this plot.
    plt.show()
    # make space for the next 
    print('')
    
    # same code, no need for code comments. Just replace the x value and name accordingly, as well as set grid.
    r, p = stats.pearsonr(train.cwtfp, train.both_sexes)
    
    train.plot.scatter('cwtfp', 'both_sexes')
    plt.title('Welfare-relevant TFP levels at Current PPP: Correlation to Suicide Rate')
    plt.text(.9, 40, f'r = {r:.3f}')
    print(f' P is less than alpha: {p < α}')
    plt.show()
    print('')
    
    r, p = stats.pearsonr(train.rwtfpna, train.both_sexes)
    
    train.plot.scatter('rwtfpna', 'both_sexes')
    plt.title('Welfare-relevant TFP at constant national prices: Correlation to Suicide Rate')
    plt.text(1.0, 40, f'r = {r:.3f}')
    print(f' P is less than alpha: {p < α}')
    plt.show()
    print('')
    
    r, p = stats.pearsonr(train.labsh, train.both_sexes)
    
    train.plot.scatter('labsh', 'both_sexes')
    plt.title('Share of labour compensation in GDP at current national prices')
    plt.text(.63, 40, f'r = {r:.3f}')
    print(f' P is less than alpha: {p < α}')
    plt.show()
    
def test_lithuania(train):
    alpha = 0.5
    print('Conducting one-sample t-test on Lithuania')
    lithuania = train[train.country == 'Lithuania'].both_sexes
    overall_mean = train.both_sexes.mean()
    
    print(f'The mean suicide rate of all countries is: {overall_mean}')
    
    t, p = stats.ttest_1samp(lithuania, overall_mean)
    
    print(f't ~ {t:.03}, p ~ {p/2:.03}, alpha = {alpha} \n')
    
    if p/2 > alpha:
        print("We fail to reject the Null Hypothesis")
    elif t < 0:
        print("We fail to reject the Null Hypothesis")
    else:
        print("We reject the Null Hypothesis")
        
    print('''
    So although the suicide rate in Lithuania is decreasing, \n it is not doing so rapidly enough \n or substantially enough to vary significantly from the overall rate.''') 
    
def scale_data(X_train, X_validate, X_test, return_scaler=False):
    '''
    Scales the 3 data splits.
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    If return_scaler is true, the scaler object will be returned as well.
    Target is not scaled.
    columns_to_scale was originally used to check whether los_angeles and orange would cause trouble
    '''
    columns_to_scale = X_train.columns
    
    X_train_scaled = X_train.copy()
    X_validate_scaled = X_validate.copy()
    X_test_scaled = X_test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(X_train_scaled[columns_to_scale])
    
    X_train_scaled[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
    X_validate_scaled[columns_to_scale] = scaler.transform(X_validate[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    if return_scaler:
        return scaler, X_train_scaled, X_validate_scaled, X_test_scaled
    else:
        return X_train_scaled, X_validate_scaled, X_test_scaled
    
def select_rfe(X_train_scaled, y_train, k, return_rankings=False, model=LinearRegression()):
    # Use the passed model, LinearRegression by default
    rfe = RFE(model, n_features_to_select=k)
     # fit the data using RFE
    rfe.fit(X_train_scaled, y_train)
    # transforming data using RFE
    X_rfe = rfe.fit_transform(X_train_scaled, y_train)
    # get mask of columns selected as list
    feature_mask = X_train_scaled.columns[rfe.support_].tolist()
    if return_rankings:
        rankings = pd.Series(dict(zip(X_train_scaled.columns, rfe.ranking_)))
        return feature_mask, rankings, X_rfe
    else:
        return feature_mask, X_rfe
    
def all_models(): 
    train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = get.wrangle_df2('both_sexes')
    X_train_scaled, X_validate_scaled, X_test_scaled = get.scale_data(X_train, X_validate, X_test)
    feature_mask = ['ctfp', 'cwtfp', 'rwtfpna', 'labsh', 'pl_con', 'pl_gdpo', 'pl_c', 'pl_g']
    baseline = pd.DataFrame({
            'median' : [y_train.median()] * y_train.size,
            'mean' : [y_train.mean()] * y_train.size
        })

    median_rmse = mod._RMSE(y_train, baseline["median"])
    mean_rmse = mod._RMSE(y_train, baseline["mean"])
    baseline_val = pd.Series([y_train.mean()] * y_validate.size)

    results = {
        'baseline' : {
            'RMSE' : mean_squared_error(y_train, baseline['mean'], squared = False),
            'R^2' : r2_score(y_train, baseline['mean']),
            'RMSE_val' : mean_squared_error(y_validate, baseline_val, squared = False),
            'R^2_val' : r2_score(y_validate, baseline_val)
        }
    }
    results
    
    model = LinearRegression()
    model.fit(X_train[feature_mask], y_train)

    r2_score(y_train, model.predict(X_train[feature_mask]))
    mean_squared_error(y_train, model.predict(X_train[feature_mask]), squared = False)
    results['linear_regression'] = {
    'RMSE' : mean_squared_error(y_train, model.predict(X_train[feature_mask]), squared = False),
    'R^2' : r2_score(y_train, model.predict(X_train[feature_mask])),
    'RMSE_val' : mean_squared_error(y_validate, model.predict(X_validate[feature_mask]), squared = False),
    'R^2_val' : r2_score(y_validate, model.predict(X_validate[feature_mask]))
    }
    
    # OLS
    model = LinearRegression(normalize=True)
    model.fit(X_train[feature_mask], y_train)

    r2_score(y_train, model.predict(X_train[feature_mask]))
    
    mean_squared_error(y_train, model.predict(X_train[feature_mask]), squared = False)
    
    results['linear_regression_OLS'] = {
    'RMSE' : mean_squared_error(y_train, model.predict(X_train[feature_mask]), squared = False),
    'R^2' : r2_score(y_train, model.predict(X_train[feature_mask])),
    'RMSE_val' : mean_squared_error(y_validate, model.predict(X_validate[feature_mask]), squared = False),
    'R^2_val' : r2_score(y_validate, model.predict(X_validate[feature_mask]))
    }
    
    # Polynomial Regression
    poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = False)
    poly.fit(X_train[feature_mask])

    X_train_poly = pd.DataFrame(
        poly.transform(X_train[feature_mask]),
        columns = poly.get_feature_names(X_train[feature_mask].columns),
        index = X_train[feature_mask].index
    )
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    r2_score(y_train, model.predict(X_train_poly))
    
    mean_squared_error(y_train, model.predict(X_train_poly), squared = False)
    
    # Before we make predictions on validate we need to prepare the X_validate set with the polynomial features.
    poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = False)
    poly.fit(X_validate[feature_mask])
    
    X_validate_poly = pd.DataFrame(
        poly.transform(X_validate[feature_mask]),
        columns = poly.get_feature_names(X_validate[feature_mask].columns),
        index = X_validate[feature_mask].index
    )
    
    results['polynomial_regression'] = {
    'RMSE' : mean_squared_error(y_train, model.predict(X_train_poly), squared = False),
    'R^2' : r2_score(y_train, model.predict(X_train_poly)),
    'RMSE_val' : mean_squared_error(y_validate, model.predict(X_validate_poly), squared = False),
    'R^2_val' : r2_score(y_validate, model.predict(X_validate_poly))
    }
    
    # Polynomial Regression with Interactions Only
    poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = True)
    poly.fit(X_train[feature_mask])
    
    X_train_poly = pd.DataFrame(
        poly.transform(X_train[feature_mask]),
        columns = poly.get_feature_names(X_train[feature_mask].columns),
        index = X_train[feature_mask].index
    )
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    r2_score(y_train, model.predict(X_train_poly))
    
    mean_squared_error(y_train, model.predict(X_train_poly), squared = False)
    
    # We have to prepare the validate set again

    poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = True)
    poly.fit(X_validate[feature_mask])
    
    X_validate_poly = pd.DataFrame(
        poly.transform(X_validate[feature_mask]),
        columns = poly.get_feature_names(X_validate[feature_mask].columns),
        index = X_validate[feature_mask].index
    )
    
    results['polynomial_regression_interactions_only'] = {
    'RMSE' : mean_squared_error(y_train, model.predict(X_train_poly), squared = False),
    'R^2' : r2_score(y_train, model.predict(X_train_poly)),
    'RMSE_val' : mean_squared_error(y_validate, model.predict(X_validate_poly), squared = False),
    'R^2_val' : r2_score(y_validate, model.predict(X_validate_poly))
    }
    
    # Tweedie with Power of 1: Poisson distribution
    # No collinearities from RFE is assumed, so alpha 0

    model = TweedieRegressor(power = 1, alpha = 0)
    model.fit(X_train[feature_mask], y_train)

    r2_score(y_train, model.predict(X_train[feature_mask]))
    
    mean_squared_error(y_train, model.predict(X_train[feature_mask]), squared = False)
    
    results['tweedie_regressor_poisson'] = {
    'RMSE' : mean_squared_error(y_train, model.predict(X_train[feature_mask]), squared = False),
    'R^2' : r2_score(y_train, model.predict(X_train[feature_mask])),
    'RMSE_val' : mean_squared_error(y_validate, model.predict(X_validate[feature_mask]), squared = False),
    'R^2_val' : r2_score(y_validate, model.predict(X_validate[feature_mask]))
    }
    
    # Tweedie with Power of 1.5: Compound Poisson Gamma distribution
    # No collinearities from RFE is assumed, so alpha 0

    model = TweedieRegressor(power = 1.5, alpha = 0)
    model.fit(X_train[feature_mask], y_train)

    r2_score(y_train, model.predict(X_train[feature_mask]))
    
    mean_squared_error(y_train, model.predict(X_train[feature_mask]), squared = False)
    
    results['tweedie_regressor_compound_poisson_gamma'] = {
    'RMSE' : mean_squared_error(y_train, model.predict(X_train[feature_mask]), squared = False),
    'R^2' : r2_score(y_train, model.predict(X_train[feature_mask])),
    'RMSE_val' : mean_squared_error(y_validate, model.predict(X_validate[feature_mask]), squared = False),
    'R^2_val' : r2_score(y_validate, model.predict(X_validate[feature_mask]))
    }
    
    return results

# Keeping for my own records, but random selection of each is a bad idea when there's value gained from the time element. 
# def split_suicide(df):
#     
#     # train/validate/test split
#     train_validate, test = train_test_split(df, test_size=.2, random_state=123)
#     train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
#     
#     print(train.shape, validate.shape, test.shape)
#     
#     
#     return train, validate, test
#

#obsolete version, removed my ability to view year directly, without consideration of the index
#def create_time_df(df):
#    '''
#    Takes in the suicide dataframe from prepare, creates a copy
#    and then converts the year column from int to date-time format,
#    with the unique year with hours timestamp set as the index.
#    '''
#    df2 = df.copy()
#    
#    # A particular approach to creating date time while avoiding pandas default to 1970
#    df2['year'] = pd.to_datetime(df2['year'], format = "%Y").dt.strftime('%Y')
#    df2.year = pd.to_datetime(df2.year, infer_datetime_format = True)
#    # add an hour to each recurrent year. 
#    df2['year'] = df2['year'] + pd.to_timedelta(df2.groupby('year').cumcount(), unit='h')
#    # set the index and sort it by date
#    df2 = df2.set_index('year').sort_index()
#    
#    return df2