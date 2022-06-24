import pandas as pd

def get_day_of_week(date, output_string = False):
    if output_string == False:
        return pd.Timestamp(date).isoweekday()
    else:
        return pd.Timestamp(date).day_name()