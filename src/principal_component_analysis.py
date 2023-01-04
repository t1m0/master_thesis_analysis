
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def principal_component_analysis(df, feature_keys):
    # https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn 
    feature_df = df[feature_keys]
    x = feature_df.values
    x = StandardScaler().fit_transform(x) # normalizing the features
    normalised_df = pd.DataFrame(x,columns=feature_keys)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(normalised_df)
    return pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])

def plot_principal_component_analysis(source_df,pca_df,target_column,targets):
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    plt.title("Principal Component Analysis",fontsize=20)
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = source_df[target_column] == target
        plt.scatter(pca_df.loc[indicesToKeep, 'principal component 1']
                , pca_df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

    plt.legend(targets,prop={'size': 15})
