import matplotlib.pyplot as plt
from src.accelerometer import plot_acceleration

def _plot_stroop_clicks(df):
    for click in df['click_number'].unique():
        click_df = df[df['click_number']==click]
        plt.axvline(x = click_df['duration'].max(), color = 'b')

def plot_stroop_stacceleration(df, title):
    plot_acceleration(df,subplots=False,title=title,additional_plotting=_plot_stroop_clicks)