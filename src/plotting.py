

import matplotlib.pyplot as plt

from src.pandas_util import get_min_value_across_columns, get_max_value_across_columns, drop_outliers_of_columns

def _box_plot_columns_single(df, class_key, columns, show_column_in_label):
    
    df_new = drop_outliers_of_columns(df,columns)
    
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(30, 7.5)

    box_plot_data = {}

    for column in columns:
        if df_new[column].dtype in [float,int]:
            if class_key != '':
                for class_value in df_new[class_key].unique():
                    
                    label = f'{column} {class_value}' if show_column_in_label else class_value
                    
                    values = df_new[df_new[class_key]==class_value][column]
                    box_plot_data[label] = values
            else:
                box_plot_data[column] = df_new[column]
        else:
            print(f"Column {column} not numeric, therefore no boxplot created!!")
    if len(box_plot_data) > 0:
        plt.boxplot(list(box_plot_data.values()),labels=list(box_plot_data.keys()))
        plt.title(columns)
        plt.tight_layout()
        plt.show()

def box_plot_columns(df, class_key='', columns=['x', 'y', 'z', 'mag'],show_column_in_label=True):
    if type(df) is list or type(df) is set:
        for single_df in df:
            _box_plot_columns_single(single_df,class_key,columns, show_column_in_label)
    else:
        _box_plot_columns_single(df,class_key,columns, show_column_in_label)