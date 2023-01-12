import pandas as pd

def _get_x_value_across_columns(df,function, fields, lower_padding, upper_padding):
    if type(df) == pd.core.groupby.generic.DataFrameGroupBy:
        final_df = df.apply(lambda x: x) 
    else:
        final_df = df

    values = []
    for field in fields:
        column_values = final_df[field]
        values.append(function(column_values))
    
    value = function(values)
    
    if value < 0:
        value = value * lower_padding
    else:
        value = value * upper_padding

    return value

def get_min_value_across_columns(df, fields=[],padding=0):

    lower_padding = 1+padding
    upper_padding = 1-padding
    
    return _get_x_value_across_columns(df,min,fields=fields,lower_padding=lower_padding, upper_padding=upper_padding)

def get_max_value_across_columns(df, fields=[],padding=0):
    
    lower_padding = 1-padding
    upper_padding = 1+padding
    
    return _get_x_value_across_columns(df,max,fields=fields,lower_padding=lower_padding, upper_padding=upper_padding)

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