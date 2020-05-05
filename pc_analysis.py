from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt


def pc_analysis(df):
    '''
    The code for this function has been adapted from:
    https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

    This function performs Principal Component Analysis (PCA) on a given wine
    dataset with 11 features (x data) and 1 target [quality] (y data). For
    thoroughness, PCA was conducted with N = [1:11] principal components to
    demonstrate how much variance is captured with N principal components.

    ** Parameters **

        df: *pandas.DataFrame*
            raw dataframe of wine input data; 12 columns: 11 features 1 target

    ** Returns **

        pc_dict: *dictionary* {'# Principal Components': [PCA object,
                                                          PC_df,
                                                          target_df]}
            dictionary of PCA object data (used for variance), df of PCs, and
            df of wine quality
    '''

    pc_dict = {}  # dict of PCAs by the number of components included

    # perform PCA for numbers of components up to the number of features
    for components in range(1, len(list(df))):

        n = components  # number of principal components
        components_string = '{0} Principal Component'.format(n)

        features = list(df)
        features.remove('quality')
        x = df.loc[:, features].values
        # y = df.loc[:, ['quality']].values

        standardized = StandardScaler().fit_transform(x)
        # x_st = pd.DataFrame(data=standardized, columns=features)

        pca = PCA(n_components=n)
        principalComponents = pca.fit_transform(standardized)

        cols = ['PC' + str(i) for i in range(1, n + 1)]
        final_df_x = pd.DataFrame(data=principalComponents,
                                  columns=cols
                                  )

        final_df_y = df[['quality']]

        pc_dict[components_string] = [pca, final_df_x, final_df_y]

    return pc_dict

# THIS PART IS FOR PLOTTING THAT I INTENTIONALLY LEFT OUT BECAUSE IT ONLY WORKS
# FOR 2 OR FEWER PRINCIPAL COMPONENTS WHICH IS PRETTY USELESS FOR THIS PROJECT
# BECAUSE I GO UP TO 11


#     # this only works if number of PCs < 2 (LOL)
#     if plot:
#         plot_pca(df, final_df)


# def plot_pca(df, final_df):

#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_xlabel('PC1', fontsize=15)
#     ax.set_ylabel('PC2', fontsize=15)
#     ax.set_title('2 component PCA', fontsize=20)
#     targets = [str(i)
#                for i in range(min(df['quality']), max(df['quality'] + 1))]
#     colors = ['r', 'g', 'b', 'c', 'm', 'y']
#     for target, color in zip(targets, colors):
#         indicesToKeep = final_df['quality'] == int(target)
#         ax.scatter(final_df.loc[indicesToKeep, 'PC1'],
#                    final_df.loc[indicesToKeep, 'PC2'], c=color, s=50)
#     ax.legend(targets)
#     ax.grid()
#     plt.show()
