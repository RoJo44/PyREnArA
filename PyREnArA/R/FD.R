### TITLE: PyREnArA (FD) ###

### AUTHORS: Florian Linsel (Institute of Computer Science, Martin-Luther-Universität Halle-Wittenberg, Von-Seckendorff-Platz 1, 06120 Halle)



# install.packages(c('geodiv','VoxR','raster','geodiv','dplyr','tidyr','imager','conflicted'))

# load required packages
require (raster)
require(geodiv)
require(dplyr)
require(tidyr)
require (imager)
require(conflicted)



# import image and transflated to dataframe
image_to_df <- function (png_path) {
  
  i = load.image(png_path)
  data = as.data.frame(i)
  data = data[,0:3] # cropping to the first three columns due to the limitation of the fractal analysis
  
  data = as.data.frame(lapply(data, as.numeric)) # translate 
  
  return (data)
  
}

# innital idea of box-counting following Benoit Mandelbrot
frac_dim <- function (png_path) {
 
  data_temp <- image_to_df (png_path)
  
  #- box counting
  FD = VoxR::box_counting(data_temp,store_fit = TRUE)
  
  # the fractal dimension and the box counting-functional result is a parameter, where the scales is set in a relation to the amount of pixels.
  # To preserve the linearity and the explanatory weight of this method we decided to exclude all scales, where the amount of pixels does not 
  #increase while decreasing the scale. The lowest scale, for which the amount of pixels still increases, we have to include into our study.
  # Due to the aforementioned linearity, we can apply different ranges of scales. 
  
  # detect and count the scales where the maximum amount of pixels are reached
  dat <- FD$fit_table %>% 
    count(N) %>% 
    dplyr::filter(n == 1) %>% 
    dplyr::select(-n)

  FD_temp <- as.data.frame(FD$fit_table) [1:length(dat$N)+1,] # select the scales which have not reached the maximum pixel amount
  
  FD_real <- lm(formula = log(FD_temp$N) ~ log(1/FD_temp$res)) # Calculating the Fractal Dimension
  
  FD_interc <- FD_real$coefficients[1] # Intercept
  FD_gradient <- FD_real$coefficients[2] # Fractal Dimension
  
  return (as.numeric(FD_gradient)) # returning the FD
  
}

frac_data <- function (png_path){ # ideal for later analysis of images not outlines
  
  data <- as.matrix(raster(png_path))

  # Calculates the 3D fractal dimension of a raster using the
  #' triangular prism surface area method.
  sfd <- (sfd(data)) 
  
  # apply metrics
  sa <- (sa(data)) # average surface roughness
  #> [1] 0.0442945
  svk <- (svk(data)) # reduced valley depth
  #> [1] 0.5449867
  ssc <- (ssc(data)) # mean summit curvature
  #> [1] -0.02192238
  
  
  FD_list <- c(sfd,sa,svk,ssc) # combine to one vector
  
  return (FD_list)
}

# based on VoxR::box_counting
fractal_dimension <- function (lf_total, dir, data_name) {
  
  data <- list() # preparation of resulting list 
  
  data <- append(data,lapply(lf_total,frac_dim)) # appending of resulting values
  
  data_2 <- data # copying the list 
  
  data_2 <- do.call(rbind.data.frame, data) # converting list to data.frame
  
  names(data_2)[names(data_2[1]) == names(data_2[1])] <- 'FD' # renaming column to FD 
  
  add_fract(data_2,dir,data_name)
  
  return (data_2)
}

add_fract <- function (x,dir,data_name) {
  
  data <- read.csv(paste0(data_name, '.csv'), sep=',', encoding='UTF-8')
  print(str(data))
  data <- cbind(data, x)
    
  write.csv(data,paste0(data_name,'_FD.csv'), row.names = FALSE)
    
}

# based on geodiv's fractal functions
fractal_stats <- function (x) {
  
  data_data <- list() # empty list for attaching data
  
  data_data <- append(data_data,lapply(x,frac_data)) # appending geodiv's data including the Fractal Dimension 
  
  data_data <- do.call(rbind.data.frame,data_data)
  
  names(data_data) <- c('FD','sa','svk','ssc')
  
  return (data_data)
  
}

filtering <- function (datfram,dir,filter_path,x,save_path){

  print(getwd())
  print(filter_path)
  filter_list <- read.csv(filter_path,encoding = 'unicode_escape')
  print(filter_list)
  datfram <- cbind(datfram, x)
  
  for (i in 1:(nrow(filter_list))) {
    #print(datfram[datfram[filter_list[2,1]]!= filter_list[2,2],])
    print(i)
    print(filter_list[i,1])
    print(filter_list[i,2])
    datfram <- datfram[datfram[filter_list[i,1]] != filter_list[i,2],]
    
    datfram <- datfram %>% drop_na()
    print(class(datfram))
    print(head(datfram))
  }
  
  write.csv(datfram,paste0(dir,save_path), row.names = FALSE,fileEncoding="UTF-8")
  
  return (datfram)
}