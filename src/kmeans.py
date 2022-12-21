from sklearn.cluster import KMeans
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random
import numpy as np
import sys
import numpy as np




def kmeans(df, feature_keys):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    kmeans = KMeans(n_clusters=2).fit(features)
    return kmeans
