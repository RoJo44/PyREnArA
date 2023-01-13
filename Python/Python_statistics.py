### TITLE: PyREnArA (statistical functions in python) ###

### AUTHOR: Robin John (Institute of Prehistoric Archeology, University of Cologne, Bernhard-Feilchenfeld-Str. 11, 50969 Köln, Germany)



# import required packages
import statistics
import subprocess
import seaborn as sns; sns.set_theme(color_codes=True)
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import svd
from PIL import Image
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import pearsonr

from ipynb.fs.full.basic_functions import write_as_csv


    
def metrics_vs_plot(directory_to_metrics, directory_to_save, parameter):
    
    df = pd.read_csv(directory_to_metrics + '.csv')
    df = df.sort_values(by = ['ka_cal_BP'], ascending = True)

    for col in df.columns:
        
        if col == 'ID' or col == 'site' or col == 'ka_cal_BP' or col == 'long' or col == 'lat':
            continue
            
        if col == 'region' or col == 'typology':
            sns.set(rc = {'figure.figsize': (24,12)}, context = {'lines.linewidth': 2}, font = 'Times New Roman', style = 'whitegrid')

            boxplot = sns.boxplot(x = parameter, y = col, data = df, color = 'grey', width = 0.2)

            xlabel = parameter.replace('_', ' ')
            ylabel = col.replace('_', ' ')

            boxplot.set_xlabel(xlabel, fontsize = 24, weight = 'bold')
            boxplot.set_ylabel(ylabel, fontsize = 24, weight = 'bold')
            boxplot.spines['left'].set_linewidth(5)
            boxplot.spines['bottom'].set_linewidth(5)
            boxplot.tick_params(labelsize = 18)

            boxplot.invert_xaxis()

            fig = boxplot.get_figure()
            fig.savefig(directory_to_save + col + '_vs_' + parameter + '.png')

            plt.close()
            
        else:
            sns.set(rc = {'figure.figsize': (24,12)}, context = {'lines.linewidth': 4}, font = 'Times New Roman', style = 'whitegrid')

            plot = sns.lineplot(x = parameter, y = col, data = df, color = 'grey')
            
            xlabel = parameter.replace('_', ' ')
            ylabel = col.replace('_', ' ') + ' in mm'
            
            plot.set_xlabel(xlabel, fontsize = 24, weight = 'bold')
            plot.set_ylabel(ylabel, fontsize = 24, weight = 'bold')
            plot.spines['left'].set_linewidth(5)
            plot.spines['bottom'].set_linewidth(5)
            plot.tick_params(labelsize = 18)
            
            plot.invert_xaxis()

            fig = plot.get_figure()
            fig.savefig(directory_to_save + col + '_vs_' + parameter + '.png')
            
            plt.close()

def standard_deviation(directory_to_data, directory_save, parameter):
    
    df = pd.read_csv(directory_to_data + '.csv')
    
    # sort by time
    df = df.sort_values(by = ['ka_cal_BP'], ascending = True)
    
    # exclude parameters like time and coordinates from analyse
    if df.columns[:1] == 'long' or df.columns[:1] == 'lat':
        df = (df.loc[:, df.columns != 'long'])
        df = (df.loc[:, df.columns != 'lat'])
    if parameter != 'ka_cal_BP':
        df = (df.loc[:, df.columns != 'ka_cal_BP'])
    
    # group data by parameter
    groupedby = df.groupby(df[parameter])
    
    # calculate standard deviation 
    grouped_std = groupedby.std(numeric_only = True)
    
    # print to csv
    grouped_std.to_csv(('{}/std_metrics_' + parameter + '.csv').format(directory_save), index=True)
    
    for col in grouped_std.columns:

            sns.set(rc = {'figure.figsize': (24,12)}, context = {'lines.linewidth': 4}, font = 'Times New Roman', style = 'whitegrid')

            plot = sns.lineplot(grouped_std, x = parameter, y = col, color = 'grey')
            
            xlabel = parameter.replace('_', ' ')
            ylabel = 'std ' + col.replace('_', ' ')
            
            plot.set_xlabel(xlabel, fontsize = 24, weight = 'bold')
            plot.set_ylabel(ylabel, fontsize = 24, weight = 'bold')
            plot.spines['left'].set_linewidth(5)
            plot.spines['bottom'].set_linewidth(5)
            plot.tick_params(labelsize = 18)
            
            plot.invert_xaxis()
            
            fig = plot.get_figure()
            fig.savefig(directory_save + 'std_' + col + '_' + str(parameter) + '.png')
            
            plt.close()
    
def descriptive_statistics_metrics(directory_to_metrics, directory_save, parameter):
    
    df = pd.read_csv(directory_to_metrics + '.csv')
   
    # exclude parameters like time and coordinates from analyse
    if df.columns[:1] == 'long' or df.columns[:1] == 'lat':
        df = (df.loc[:, df.columns != 'long'])
        df = (df.loc[:, df.columns != 'lat'])
    if parameter != 'ka_cal_BP':
        df = (df.loc[:, df.columns != 'ka_cal_BP'])

    # group data by parameter
    groupedby = df.groupby(df[parameter])

    # descriptive statistics of metrical data
    grouped_data_analyse = groupedby.describe()

    grouped_data_analyse.to_csv(('{}/descriptive_statistics_metrics_' + parameter + '.csv').format(directory_save), index=True)
    
def specific_mean_metrics(directory_to_metrics, directory_save, parameter):
    
    df = pd.read_csv(directory_to_metrics + '.csv')

    # exclude parameters like time and coordinates from analyse
    if df.columns[:1] == 'long' or df.columns[:1] == 'lat':
        df = (df.loc[:, df.columns != 'long'])
        df = (df.loc[:, df.columns != 'lat'])
    if parameter != 'ka_cal_BP':
        df = (df.loc[:, df.columns != 'ka_cal_BP'])
        
    # group data by parameter
    groupedby = df.groupby(df[parameter])
    
    grouped_mean = groupedby.mean(numeric_only = True)
    
    grouped_mean.to_csv(('{}/' + parameter + '_specific_mean_metrics.csv').format(directory_save), index=True)

def pearsonr_pval(x,y):
    
    return pearsonr(x,y)[1]
    
def pearson_correlation(directory_to_metrics, directory_save, parameter):
    
    df = pd.read_csv(directory_to_metrics + '.csv')
    
    correlation = ['r']
    significance = ['p']
    metric = ['metric']
    tested_against = [str(parameter)]
    
    for (colname, colval) in df.items():

        if colname == 'ID' or colname == 'ka_cal_BP' or colname == 'site' or colname == 'typology' or colname == 'region' or colname == 'long' or colname == 'lat':
            continue
        else:
            correl = df[colname].corr(df[parameter])
            signi = df[colname].corr(df[parameter], method = pearsonr_pval)
            
            metric.append(colname)
            tested_against.append(parameter)
            correlation.append(correl)
            significance.append(signi)
        
    metrics_correlation = zip(metric, tested_against, correlation, significance)
    write_as_csv(metrics_correlation, directory_save + 'correlation_metrics_' + str(parameter))

def pairplot_pearson(directory_to_metrics, directory_save):
    
    df = pd.read_csv(directory_to_metrics + '.csv')
    
    df = (df.loc[:, df.columns != 'long'])
    df = (df.loc[:, df.columns != 'lat'])
    
    df = df.sort_values(by = ['ka_cal_BP'], ascending = True)
       
    df = (df.loc[:, df.columns != 'ka_cal_BP'])
    
    sns.set(font = 'Times New Roman', font_scale = 2)

    pairplot = sns.pairplot(data = df, hue = 'site', palette = 'magma_r')
    sns.move_legend(pairplot, "upper left", bbox_to_anchor = (0.92, 0.98))
    
    (xmin, _), (_, ymax) = pairplot.axes[0, 0].get_position().get_points()
    (_, ymin), (xmax, _) = pairplot.axes[-1, -1].get_position().get_points()

    ax = pairplot.fig.add_axes([xmin, ymin, xmax - xmin, ymax - ymin], facecolor = 'none')

    corr = df.corr(numeric_only = True)
    mask = np.tril(np.ones_like(corr, dtype = bool))
    sns.heatmap(corr, mask = mask, cmap = 'coolwarm', vmax = 0.9, vmin = -.9, linewidths = 30, cbar = False, annot = True, annot_kws = {'size': 20}, ax = ax)

    ax.set_xticks([])
    ax.set_yticks([])
    
    pairplot.savefig(directory_save + 'pairplot_pearson.png')
    
    plt.close()

def principal_component_analysis(directory_to_data, directory_save, style = 'site', sorting = 'ka_cal_BP', features = ['width', 'length', 'l_w_index', 'area', 'percent_area', 'contour_length', 'area_contour_ratio', 'upper_to_lower_ratio', 'left_to_right_ratio', 'MP_CM_x_offset', 'MP_CM_y_offset']):
    
    df = pd.read_csv(directory_to_data + '.csv')
    
    # set dependend variables
    x = df.loc[:, features].values
    
    # standardise features
    x = StandardScaler().fit_transform(x)
    
    # perform PCA  
    pcamodel = PCA(n_components = 5)
    
    # screeplot
    pca_scree = pcamodel.fit_transform(x)

    sns.set(rc = {'figure.figsize': (20, 10)}, context = {'lines.linewidth': 4}, font = 'Times New Roman', style = 'whitegrid')
    screeplot = sns.barplot(x = np.arange(1, len(pcamodel.explained_variance_ratio_) + 1), y = pcamodel.explained_variance_ratio_, color = 'gray')
    
    screeplot.set_ylabel('Explained variance', fontsize = 24, weight = 'bold')
    screeplot.set_xlabel('Components', fontsize = 24, weight = 'bold')
    screeplot.spines['left'].set_linewidth(5)
    screeplot.spines['bottom'].set_linewidth(5)
    screeplot.tick_params(labelsize = 18)

    fig_scree = screeplot.get_figure()
    fig_scree.savefig(directory_save + 'PCA_screeplot.png')
    
    plt.close()
    
    # scatterplot
    pcamodel_2 = PCA(n_components = 2)
    pca_scatter = pcamodel_2.fit_transform(x)
    principalDf = pd.DataFrame(data = pca_scatter, columns = ['principal component 1', 'principal component 2'])
    
    finalDf = pd.concat([principalDf, df[style], df[sorting]], axis = 1)
    finalDf = finalDf.sort_values(by = sorting, ascending = True)
    
    PC1_val = np.round(pcamodel_2.explained_variance_ratio_[0], 5)
    PC2_val = np.round(pcamodel_2.explained_variance_ratio_[1], 5)
  
    sns.set(rc = {'figure.figsize': (20, 10)}, context = {'lines.linewidth': 4}, font = 'Times New Roman', style = 'whitegrid')
    scatterplot = sns.scatterplot(x = finalDf['principal component 1'], y = finalDf['principal component 2'], data = finalDf, hue = style, palette = 'magma_r', s = 150)
    
    scatterplot.set_xlabel('PC 1 ' + '(' + str(PC1_val) + ')', fontsize = 24, weight = 'bold')
    scatterplot.set_ylabel('PC 2 ' + '(' + str(PC2_val) + ')', fontsize = 24, weight = 'bold')
    scatterplot.spines['left'].set_linewidth(5)
    scatterplot.spines['bottom'].set_linewidth(5)
    scatterplot.tick_params(labelsize = 18)
            
    fig_scatter = scatterplot.get_figure()
    fig_scatter.savefig(directory_save + 'PCA_plot.png')
                
    plt.close()
    
    # plot effect of variables on each components
    sns.set(rc = {'figure.figsize': (20, 12)}, context = {'lines.linewidth': 4}, font = 'Times New Roman')
    x = pd.DataFrame(x, columns = features)
    
    xtick_params = list(x.columns)

    #effect = pcamodel.components_(numeric_only = True)    
    effectplot = sns.heatmap(pcamodel.components_, cmap = 'coolwarm', yticklabels = [str(x) for x in range(1, pcamodel.n_components_ + 1)], xticklabels = xtick_params, annot = True, cbar = False)
    
    effectplot.set_ylabel('Components', fontsize = 24, weight = 'bold')
    effectplot.tick_params(labelsize = 18)
    effectplot.set_aspect("equal")
    plt.xticks(rotation = 90)
    
    fig_effect = effectplot.get_figure()
    fig_effect.savefig(directory_save + 'PCA_variable_effects.png')
                
    plt.close()
    
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components of PCA): %s' % str(pcamodel.explained_variance_ratio_[0]) + str(pcamodel.explained_variance_ratio_[1]))

def dendro_cluster_plot(directory_to_data, directory_save, index = None, parameter = 'site', sorting = 'ka_cal_BP', features = ['width', 'length', 'l_w_index', 'area', 'percent_area', 'contour_length', 'area_contour_ratio', 'upper_to_lower_ratio', 'left_to_right_ratio', 'tip_angle', 'MP_CM_x_offset', 'MP_CM_y_offset']):
    
    df = pd.read_csv(directory_to_data + '.csv', index_col = index)
    df = df.sort_values(by = [sorting], ascending = True)
    numerical_cols = df[df.columns.intersection(features)]
    
    # set cluster column
    lut = dict(zip(df[parameter].unique(), sns.color_palette(palette = 'magma_r', n_colors = len(set(df[parameter])))))
    colors = df[parameter].map(lut)
    
    # plot
    sns.set(context = {'lines.linewidth': 2}, font = 'Times New Roman')
    dendro_plot = sns.clustermap(numerical_cols, standard_scale = 1, row_colors = colors, cmap = 'magma', figsize = (10, 10), cbar_pos = (0, 0.825, 0.05, 0.15))

    dendro_plot.savefig(directory_save + 'dendro_cluster_plot.png')
    
    plt.close()
    
def manova(directory_to_data, directory_save, explanatory_variable = 'ka_cal_BP', features = ['width', 'length', 'l_w_index', 'area', 'percent_area', 'contour_length', 'area_contour_ratio', 'upper_to_lower_ratio', 'left_to_right_ratio', 'tip_angle', 'MP_CM_x_offset', 'MP_CM_y_offset']):

    df = pd.read_csv(directory_to_data)
    
    depend_var = df[df.columns.intersection(features)]
    independ_var = df[explanatory_variable]
    
    manova = MANOVA.from_formula('width + length + l_w_index + area + percent_area + contour_length + area_contour_ratio + upper_to_lower_ratio +left_to_right_ratio + tip_angle + MP_CM_x_offset + MP_CM_y_offset~ka_cal_BP', data=df)
    
    manova_results = manova.mv_test()
    
    print('Manova results: ')
    print(manova_results)
    
    write_as_csv(manova_results, directory_save + 'manova_' + str(explanatory_variable))

def linear_discriminant_analysis(directory_to_data, directory_save, style = 'site', explanatory_variable = 'ka_cal_BP', features = ['width', 'length', 'l_w_index', 'area', 'percent_area', 'contour_length', 'area_contour_ratio', 'upper_to_lower_ratio', 'left_to_right_ratio', 'tip_angle', 'MP_CM_x_offset', 'MP_CM_y_offset']):
    
    df = pd.read_csv(directory_to_data)
    
    # set dependend variables
    X = df.loc[:, features].values
    
    # set independend variable
    y = df[explanatory_variable].astype('int')
    target_names = pd.unique(df[explanatory_variable])
    
    lda = LDA()
    model = lda.fit(X, y)
    X_r = lda.fit(X, y).transform(X)
    
    #Define method to evaluate model
    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

    #evaluate model
    scores = cross_val_score(model, X, y, scoring = 'accuracy', cv =cv, n_jobs = -1)
    print('LDA model accuracy: ' + str(np.mean(scores)))
    
    X_r_df = pd.DataFrame(X_r, columns = ['1', '2', '3'])

    finalDf = pd.concat([df[style], df[explanatory_variable], X_r_df['1'], X_r_df['2']], axis = 1)
    if explanatory_variable == 'ka_cal_BP':
        finalDf = finalDf.sort_values(by = 'ka_cal_BP', ascending = True)
    
    LDA1_val = np.round(lda.explained_variance_ratio_[0], 5)
    LDA2_val = np.round(lda.explained_variance_ratio_[1], 5)
    
    sns.set(rc = {'figure.figsize': (20, 10)}, context = {'lines.linewidth': 4}, font = 'Times New Roman', style = 'whitegrid')
    plot = sns.scatterplot(x = finalDf['1'], y = finalDf['2'], data = finalDf, hue = style, palette = 'magma_r', s = 150)
    
    plot.set_xlabel('LDA 1 ' + '(' + str(LDA1_val) + ')', fontsize = 24, weight = 'bold')
    plot.set_ylabel('LDA 2 ' + '(' + str(LDA2_val) + ')', fontsize = 24, weight = 'bold')
    plot.spines['left'].set_linewidth(5)
    plot.spines['bottom'].set_linewidth(5)
    plot.tick_params(labelsize = 18)
            
    fig = plot.get_figure()
    fig.savefig(directory_save + 'LDA_plot.png')
    
    plt.close()
    
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components of LDA): %s' % str(lda.explained_variance_ratio_[0]) + ' ' + str(lda.explained_variance_ratio_[1]))