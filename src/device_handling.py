
def split_by_device(df):
    device_dfs = []
    for device in df['device'].unique():
        mask = df['device'] == device
        current_df = df[mask]
        device_dfs.append(current_df)
    return device_dfs