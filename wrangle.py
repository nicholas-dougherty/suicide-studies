import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# import scaling methods
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
# import modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

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
    df2.year = pd.to_datetime(df2.year, infer_datetime_format = True)
    # add an hour to each recurrent year. 
    df2['year'] = df2['year'] + pd.to_timedelta(df2.groupby('year').cumcount(), unit='h')
    # set the index and sort it by date
    df2 = df2.set_index('year').sort_index()
    
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
    print('Dataframe Input received: Splitting Data .5/.3/.2.')
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train_time = df[: train_size]
    validate_time = df[train_size : validate_end_index]
    test_time = df[validate_end_index : ]
    print(f'Train: {train_time.shape}, Validate {validate_time.shape}, and Test {test_time.shape} are ready.\
    \n Proceed with EDA.')
    return train_time, validate_time, test_time

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