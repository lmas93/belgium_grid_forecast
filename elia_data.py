'''Download and return historical day - 1 belgian grid data from elia '''
import pandas as pd

def get_elia_data_ods001():
    '''read csv into pandas dataframe'''
    ods001_url = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/exports/csv?lang=en&timezone=Europe%2FBrussels&use_labels=true&delimiter=%3B"
    data_raw = pd.read_csv(ods001_url, delimiter = ';')

    return data_raw
