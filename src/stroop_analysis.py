import matplotlib.pyplot as plt

def plot_stroop_stacceleration(subject_df, title):
    
    for click in subject_df['click_number'].unique():
        click_df = subject_df[subject_df['click_number']==click]
        plt.axvline(x = click_df['duration'].max(), color = 'b')
    
    plt.plot(subject_df['duration'].tolist(), subject_df['x'].tolist(), label = f"x", linestyle='solid')
    plt.plot(subject_df['duration'].tolist(), subject_df['y'].tolist(), label = f"y", linestyle='dashed')
    plt.plot(subject_df['duration'].tolist(), subject_df['z'].tolist(), label = f"z", linestyle='dotted')
    plt.plot(subject_df['duration'].tolist(), subject_df['mag'].tolist(), label = f"magnitude", linestyle='dashdot')
    
        
    plt.legend()
    plt.title(title)
    
    plt.show()