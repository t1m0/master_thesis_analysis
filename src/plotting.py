

import matplotlib.pyplot as plt

from src.pandas_util import get_min_value_across_columns, get_max_value_across_columns, drop_outliers_of_columns

def _box_plot_columns_single(df, class_key, columns):
    
    df_new = drop_outliers_of_columns(df,columns)

    min_value = get_min_value_across_columns(df_new, columns) * 0.8
    max_value = get_max_value_across_columns(df_new, columns) * 1.2

    if class_key != '':
        fig, ax = plt.subplots(1,2)
        plt.setp(ax, ylim=(min_value,max_value))
        fig.set_size_inches(30, 5)
        final_df = df_new.groupby([class_key])[columns]
        final_df.boxplot(fontsize=20, ax=ax)
    else:
        plt.ylim(min_value,max_value)
        df_new[columns].boxplot(fontsize=20)
    
    plt.show()

def box_plot_columns(df, class_key='', columns=['x', 'y', 'z', 'mag']):
    if type(df) is list or type(df) is set:
        for single_df in df:
            _box_plot_columns_single(single_df,class_key,columns)
    else:
        _box_plot_columns_single(df,class_key,columns)