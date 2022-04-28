import pandas as pd
import numpy as np
from itertools import Counter 

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

ef remove_columns(df, cols_to_remove):
    '''
    This function takes in a pandas dataframe and a list of columns to remove. It drops those columns from the original df and returns the df.
    '''
    df = df.drop(columns=cols_to_remove)
    return df
                 
                 
def handle_missing_values(df, prop_required_column=0.5 , prop_required_row=0.5):
    '''
    This function takes in a pandas dataframe, default proportion of required columns (set to 50%) and proprtion of required rows (set to 75%).
    It drops any rows or columns that contain null values more than the threshold specified from the original dataframe and returns that dataframe.
    
    Prior to returning that data, it will print statistics and list counts/names of removed columns/row counts 
    '''
    original_cols = df.columns.to_list()
    original_rows = df.shape[0]
    threshold = int(round(prop_required_column * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    remaining_cols = df.columns.to_list()
    remaining_rows = df.shape[0]
    dropped_col_count = len(original_cols) - len(remaining_cols)
    dropped_cols = list((Counter(original_cols) - Counter(remaining_cols)).elements())
    print(f'The following {dropped_col_count} columns were dropped because they were missing more than {prop_required_column * 100}% of data: \n{dropped_cols}\n')
    dropped_rows = original_rows - remaining_rows
    print(f'{dropped_rows} rows were dropped because they were missing more than {prop_required_row * 100}% of data')
    return df

# combined in one function
def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.5):
    '''
    This function calls the remove_columns and handle_missing_values to drop columns that need to be removed. It also drops rows and columns that have more 
    missing values than the specified threshold.
    '''
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df