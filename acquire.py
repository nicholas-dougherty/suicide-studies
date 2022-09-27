import numpy as np
import pandas as pd 

def acquire_data():
    df = pd.read_csv('combined.csv')
    return df 
#