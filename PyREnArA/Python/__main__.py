### TITLE: PyREnArA ###

### AUTHORs: 
# Robin John (Institute of Prehistoric Archeology, University of Cologne, Bernhard-Feilchenfeld-Str. 11, 50969 Köln, Germany),
# Florian Linsel (Institute of Computer Science, Martin-Luther-Universität Halle-Wittenberg, Von-Seckendorff-Platz 1, 06120 Halle),
# Dr. Georg Roth (Institut für Prähistorische Archäologie, Freie Universität Berlin, Fabeckstr. 23-25, 14195 Berlin, Germany),
# Dr. Andreas Maier (Institute of Prehistoric Archeology, University of Cologne, Bernhard-Feilchenfeld-Str. 11, 50969 Köln, Germany)



# import basic functions
from basic_functions import *
# import Python_statistics functions
from Python_statistics import *
# import R_statistics functions
from R_statistics import *


# enter directory for R
R_HOME = "C:/Program Files/R/R-4.2.2"
# enter working-directroy
directory = "C:/.../.../.../Gravettian_points_from_Lower_Austria_and_Moravia/"

import_R(R_HOME)

# load all images from the directory into the program and reorient them,
# store outlines of each artefact within a folder (by default named after the imagename),
# conduct metrical investigation (results are saved within the folder directory)
# (output name by default: imagename + "_metrics.csv")
combined_metrics(directory)

### merge data, FD, filtering, grouping

# merge metrical data 
# (output name by default: "data_metrics_merged.csv")
merge_metrics_data(directory, directory)

# calculate FD for all artefacts (output name by default: "data_metrics_merged_FD.csv")
FD(directory, directory + 'data_metrics_merged')

# merge metrical data (with FD) and metadata 
# (output name by default: "data_metrics_merged_metadata.csv")
merge_metrics_and_metadata(directory + 'data_metrics_merged_FD', directory + 'data_metadata', directory)

# if necessary, filter data by any parameter 
# (output name by default: "data_metrics_merged_metadata_filtered.csv")
filter_data(directory + 'data_metrics_merged_metadata', directory + 'data_filter', directory)

# generate grouped csv-datafiles 
# (output name by default name of group)
# set 'group_by' to custom group data set
grouped_csvs(directory, directory + 'data_metrics_merged_metadata_filtered', group_by = 'typology')

### statistics in Python

# create folder to save plots and statistics in 
python_statistics_folder = create_folder('Python_statistics', directory)

# create plots of metrical data ploted against parameter like time within python_statistics folder directory
# (saved within Python_statistics folder directory)
# set "parameter = '...'" to change x-axis in plots
metrics_vs_plot(directory + 'data_metrics_merged_metadata_filtered', python_statistics_folder, parameter = 'ka_cal_BP')

# descriptive statistics of metrical data (time or site as parameter)
# (saved within Python_statistics folder directory)
# (output name by default: "'descriptive_statistics_metrics_' + selected parameter + '.csv'")
# set "parameter = '...'" to change grouping of data set
descriptive_statistics_metrics(directory + 'data_metrics_merged_metadata_filtered', python_statistics_folder, parameter = 'ka_cal_BP')

# observe standard-deviations in metrics as sign of standardisation (time or site as parameter) and create plots
# (saved within Python_statistics folder directory)
# (output name by default: "'std_metrics_' + selected parameter + '.csv'")
# set "parameter = '...'" to change x-axis in plots
standard_deviation(directory + 'data_metrics_merged_metadata_filtered', python_statistics_folder, parameter = 'ka_cal_BP')

# parameter specific mean metrics
# (saved within Python_statistics folder directory)
# (output name by default: "selected parameter + '_specific_mean_metrics.csv'")
# set "parameter = '...'" to change grouping of data set
specific_mean_metrics(directory + 'tanged_point', python_statistics_folder, parameter = 'site_layer')

# calculate correlation between parameter like time and metrics and test for significance
# (saved within Python_statistics folder directory)
# (output name by default: "'correlation_metrics_' + selected parameter + '.csv'")
pearson_correlation(directory + 'data_metrics_merged_metadata_filtered', python_statistics_folder, parameter = 'ka_cal_BP')

# create pearson pairplot with all metrical data 
# (saved within Python_statistics folder directory)
pairplot_pearson(directory + 'data_metrics_merged_metadata_filtered', python_statistics_folder)

# create dendro-cluster-plot 
# (saved within Python_statistics folder directory)
# by default all metrical data is used, to specify used data set "features = ['...', '...']"
# by default 'site' is used to label cases, to specify labeling set "parameter = '...'"
# by default 'ka_cal_BP' is used to sort cases, to specify sorting set "sorting = '...'"
# to set the index (right hand side of the plot) use "index = '...'"
dendro_cluster_plot(directory + 'data_metrics_merged_metadata_filtered', python_statistics_folder)

# create PCA-plots 
# (saved within Python_statistics folder directory)
# by default all metrical data is used, to specify used data set "features = ['...', '...']"
# by default 'site' is used to label cases, to specify labeling set "style = '...'"
# by default 'ka_cal_BP' is used to sort cases, to specify sorting set "sorting = '...'"
principal_component_analysis(directory + 'data_metrics_merged_metadata', python_statistics_folder, style = 'typology')


### statistics in R

# create folder to save plots and statistics in 
R_statistics_folder = create_folder('R_statistics', directory)

# conduct GMM-functions 
# (saved within a seperate GMM folder within R_statistics folder directory)
# set "parameter = '...'" to change grouping of data set in meanshape-plot
GMM(directory, directory + 'shouldered_point', 'older_younger')

# conduct RDA function
# (saved within a seperate RDA folder within R_statistics folder directory)
RDA = RDA(directory, directory + 'data_metrics_merged_metadata_filtered_GMM_clust_2', 'ka_cal_BP')

# conduct MEM function
# (saved within a seperate MEM folder within R_statistics folder directory)
MEM(directory, directory + 'Python_statistics/' + 'site_specific_mean_metrics_clust_1')