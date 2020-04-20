from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pc_analysis(df, plot=False):

    n = 9  # number of principal components

    features = list(df)
    x = df.loc[:, features].values
    y = df.loc[:, ['quality']].values

    standardized = StandardScaler().fit_transform(x)
    x_st = pd.DataFrame(data=standardized, columns=features)

    pca = PCA(n_components=n)
    principalComponents = pca.fit_transform(standardized)

    cols = ['PC' + str(i) for i in range(1, n+1)]
    principal_df = pd.DataFrame(data=principalComponents,
                                columns=cols
                                )

    final_df = pd.concat([principal_df, df[['quality']]], axis=1)

    # print(final_df.head())
    print(sum(pca.explained_variance_ratio_))

    if plot:
        plot_pca(df, final_df)


def plot_pca(df, final_df):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [str(i)
               for i in range(min(df['quality']), max(df['quality'] + 1))]
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for target, color in zip(targets, colors):
        indicesToKeep = final_df['quality'] == int(target)
        ax.scatter(final_df.loc[indicesToKeep, 'PC1'],
                   final_df.loc[indicesToKeep, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()
