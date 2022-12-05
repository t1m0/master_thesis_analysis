def drop_outliers_of_column(df, column):
    q_max = df[column].quantile(0.90)
    q_min = df[column].quantile(0.10)
    df_new = df[df[column] < q_max]
    df_new = df_new[df_new[column] > q_min]
    return df_new
