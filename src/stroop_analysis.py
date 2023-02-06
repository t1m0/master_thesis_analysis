import matplotlib.pyplot as plt
from src.accelerometer import plot_acceleration

def _plot_stroop_clicks(df,ax=None):
    for click in df['click_number'].unique():
        click_df = df[df['click_number']==click]
        if ax == None:
            plt.axvline(x = click_df['duration'].max(), color = 'b')
        else:
            ax.axvline(x = click_df['duration'].max(), color = 'b')

def plot_stroop_stacceleration(df, title, save_to_file=False):
    plot_acceleration(df,save_to_file,title=title,additional_plotting=_plot_stroop_clicks)