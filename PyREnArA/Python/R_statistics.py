### TITLE: PyREnArA (import R-statistics functions) ###

### AUTHORS: 
# Robin John (Institute of Prehistoric Archeology, University of Cologne, Bernhard-Feilchenfeld-Str. 11, 50969 Köln, Germany),
# Florian Linsel (Institute of Computer Science, Martin-Luther-Universität Halle-Wittenberg, Von-Seckendorff-Platz 1, 06120 Halle)



# import required packages
import os
import pandas as pd

from ipynb.fs.full.basic_functions import create_folder


# function for tracing all outline images
def get_outline_dir(directory):

    outline_list = list()
        
    valid_imageformats = ['.png', '.tif']
    
    for filename in os.listdir(directory):
        
        imageformat = os.path.splitext(filename)[1]

        if imageformat.lower() in valid_imageformats:
            continue
        if filename.endswith(".csv"):
            continue
            
        name = filename
        if name == 'Python_statistics' or name == 'R_statistics':
            continue
        
        out_dir = directory + name + '/' + name + '_outlines/'
        
        for outline in os.listdir(out_dir):

            if outline.endswith(".jpg"):    
                outline_list.append(directory + name + '/' + name + '_outlines/' + outline)

    return outline_list

# imoport R to Python
def import_R(R_HOME):
    
    if not os.environ.get("R_HOME"):
        print ('Using ' + R_HOME + ' as new R_HOME variable')
        os.environ['R_HOME'] = R_HOME

# call GMM functions from R
def GMM(directory, directory_to_data, parameter):
    
    import rpy2.robjects as robjects

    r = robjects.r
    r('rm(list=(ls(all=TRUE)))')
    r.source('../R/GMM.R')
    r['options'](warn=-1)

    GMM_function = robjects.globalenv['GMM_function']

    return GMM_function(directory, directory_to_data, parameter)

# call MEM functions from R
def MEM(directory, directory_to_data):
    
    import rpy2.robjects as robjects
    
    r = robjects.r
    r.source('../R/MEM.R')
    
    MEM_function = robjects.globalenv['MEM_function']
    
    return MEM_function(directory, directory_to_data)

# call RDA functions from R
def RDA(directory, directory_to_data, parameter):
    
    import rpy2.robjects as robjects
    
    r = robjects.r
    r.source('../R/RDA.R')
    
    RDA_function = robjects.globalenv['RDA_function']
    
    return RDA_function(directory, directory_to_data, parameter)

# call Fractal Dimension function from R
def FD(directory, directory_to_data):

    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    import subprocess

    import rpy2.robjects as robjects
    r = robjects.r
    r.source('../R/FD.R')
    r['options'](warn=-1)

    from rpy2.robjects.conversion import localconverter

    fractal_dimension = robjects.globalenv['fractal_dimension']
    
    outline_list = get_outline_dir(directory)

    return fractal_dimension(outline_list, directory, directory_to_data)