import pandas as pd

def get_min_value_across_columns(df, fields=[]):
    if type(df) == pd.core.groupby.generic.DataFrameGroupBy:
        final_df = df.apply(lambda x: x) 
    else:
        final_df = df

    min_values = []
    for field in fields:
        min_values.append(final_df[field].min())
    return min(min_values)

def get_max_value_across_columns(df, fields=[]):
    if type(df) == pd.core.groupby.generic.DataFrameGroupBy:
        final_df = df.apply(lambda x: x) 
    else:
        final_df = df

    min_values = []
    for field in fields:
        min_values.append(final_df[field].max())
    return max(min_values)

def drop_outliers_of_column(df, column):
    q_max = df[column].quantile(0.90)
    q_min = df[column].quantile(0.10)
    df_new = df[df[column] < q_max]
    df_new = df_new[df_new[column] > q_min]
    return df_new
