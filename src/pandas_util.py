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

def drop_outliers_of_columns(df, columns):
    df_new = df.copy()
    for column in columns:
        df_new = drop_outliers_of_column(df_new, column, upper_quantile=0.90)
    return df_new

def drop_outliers_of_column(df, column, lower_quantile=0.10, upper_quantile=0.90):
    q_max = df[column].quantile(upper_quantile)
    q_min = df[column].quantile(lower_quantile)
    df_new = df[(df[column] <= q_max) & (df[column] >= q_min)]
    return df_new

def correlation_matrix(df, cut_off_correlation=0.8):
    correlation_matrix = df.corr()
    return correlation_matrix.style.apply(lambda x: ["background: red" if v > cut_off_correlation or v < -cut_off_correlation else "" for v in x], axis = 1).format(precision=2)