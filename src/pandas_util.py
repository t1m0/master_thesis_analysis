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
        if df_new[column].dtype in [float,int]:
            df_new = drop_outliers_of_column(df_new, column, upper_quantile=0.90)
    return df_new

def drop_outliers_of_column(df, column, lower_quantile=0.10, upper_quantile=0.90):
    q_max = df[column].quantile(upper_quantile)
    q_min = df[column].quantile(lower_quantile)
    df_new = df[(df[column] <= q_max) & (df[column] >= q_min)]
    return df_new

def _correlation_matrix_style(v, upper_cut_off_correlation, lower_cut_off_correlation):
    upper_cut_off = (v > upper_cut_off_correlation and v > 0) or (v < -upper_cut_off_correlation and v < 0)
    lower_cut_off = (v < lower_cut_off_correlation and v > 0) or (v > -lower_cut_off_correlation and v < 0)
    if upper_cut_off:
        return "background: red"
    elif lower_cut_off:
        return "background: green"
    else:
        return "background: orange"

def correlation_matrix(df, upper_cut_off_correlation=0.8,lower_cut_off_correlation=0.2):
    correlation_matrix = df.corr()
    return correlation_matrix.style.apply(lambda x: [_correlation_matrix_style(v, upper_cut_off_correlation, lower_cut_off_correlation) for v in x], axis = 1).format(precision=2)

def only_numeric_columns(df,columns = []):
    if len(columns) == 0:
        columns = df.columns
    numeric_columns = []
    for column in columns:
        if df[column].dtype in [float,int]:
            numeric_columns.append(column)
    return numeric_columns

def extract_subject_dataframe(df,age_group,subject_index=-1):
    subject = df[df['age_group']==age_group]['subject'].unique()[subject_index]
    subject_df = df[df['subject'] == subject]
    return subject_df

def extract_subject_dataframes(df,subject_index=-1):
    subject_30_df = extract_subject_dataframe(df,30,subject_index)
    subject_50_df = extract_subject_dataframe(df,50,subject_index)
    return subject_30_df, subject_50_df


def extract_sample_sessions(df,subject_index=-1,session_index=-1):
    subject_30_df, subject_50_df = extract_subject_dataframes(df, subject_index)
    uuid_30 = subject_30_df['uuid'].unique()[session_index]
    uuid_50 = subject_50_df['uuid'].unique()[session_index]
    single_session_30_df = df[df['uuid'] == uuid_30]
    single_session_50_df = df[df['uuid'] == uuid_50]
    return single_session_30_df, single_session_50_df

def split_by_column(df, column):
    device_dfs = []
    for device in df[column].unique():
        mask = df[column] == device
        current_df = df[mask]
        device_dfs.append(current_df)
    return device_dfs

def reduce_to_one_session_for_subject(df):
    df_copy = df.copy()
    uuids = []
    for subject in df_copy['subject']:
        uuid = df_copy[df_copy['subject'] == subject]['uuid'].unique().any()
        if uuid not in uuids:
            uuids.append(uuid)
    return df_copy[df_copy['uuid'].isin(uuids)]