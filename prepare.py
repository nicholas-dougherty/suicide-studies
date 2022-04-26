import pandas as pd
import numpy as np

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