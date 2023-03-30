### TITLE: PyREnArA (GMM) ###

### AUTHORS: Florian Linsel (Institute of Computer Science, Martin-Luther-Universität Halle-Wittenberg, Von-Seckendorff-Platz 1, 06120 Halle) 
### and Dr. Georg Roth (Institut für Prähistorische Archäologie, Freie Universität Berlin, Fabeckstr. 23-25, 14195 Berlin, Germany)



# load required packages
require(Momocs)
require(ggplot2)
require(cowplot)
require(cluster)
require(data.table)
require(stringr)
require(tidyverse)
require(borealis)
require(RColorBrewer)


load_images <- function (x) {

    invisible( y <- strsplit(x, "/"))

    data_name <- y[[1]][length(y[[1]])]
    invisible(lf <- list.files(paste(x,'/', data_name,'_outlines/', sep=''), full.names=TRUE, pattern='\\jpg$'))

    invisible(
        if (exists('lf_total') == TRUE)
        {
          invisible(lf_total <<- cbind(c(lf_total,lf)))
        }
        else
        {
          invisible(lf_total <<- c(lf))
        }
    )
}

GMM_filt <- function (dir,outlines.data) {

    ID <- outlines.data['ID']

    dir_list <- list.dirs(path = dir, full.names = TRUE, recursive = FALSE)

    a <- lapply(dir_list,load_images)

    outlines <- invisible(import_jpg(lf_total))

    lf <- str_split(lf_total,'/')
    lf <- data.table::transpose (lf)

    artefact <- c() 
    folder <- c()
    site <- c()

    for (i in lf_total){

        i_2 <-  str_split(i,'/')

        i_artefact <- i_2[[1]][length(i_2[[1]])]
        i_folder <- i_2[[1]][length(i_2[[1]])-1]
        i_site <- i_2[[1]][length(i_2[[1]])-2]
        artefact <- c(artefact,substr(i_artefact,1,nchar(i_artefact)-4))
        folder <- c(folder, substr(i_folder,1,nchar(i_folder)))
        site <- c(site, substr(i_site,1,nchar(i_site)))
    }
  
    filter_list <- as.data.frame(ID)

    artefact_filt <- artefact[artefact %in% filter_list$ID]
    folder_filt <- folder[artefact %in% filter_list$ID]
    site_filt <- site[artefact %in% filter_list$ID]

    # 
    lf_total_total = c()
    lf_total_total <- c(paste0(dir,paste(site_filt, folder_filt, artefact_filt, sep="/"),".jpg"))

    # image import
    b <- lapply(lf_total_total,load_images)
    outlines_filt <- invisible(import_jpg(lf_total_total))

    return_list <- list("outlines_filt" = outlines_filt, "outlines.data" = outlines.data)

    return (return_list)
}

resizePixels <- function(my_file,h_size) {

    library(magick)

    im <- image_read(paste0(my_file,'.jpg'))
    w_h_ratio <- width(im)/height(im)
    print(paste0('x',h_size))

    thmb <- image_scale(im, geometry = paste0('x',h_size))#paste0('x',h_size))#resize(im,round(h_size*w_h_ratio),round(h_size))

    thres <- image_level(thmb,black_point = 10, white_point = 10,mid_point = 100,channel = NULL) #threshold(thmb,"5.7%") %>% plot(main="Determinant: 1% highest values")

    image_write(thres,(paste0(my_file,'_res.jpg')))
}

GMM <- function (dir, outlines_filt, outlines.data_filt, data_metrics, parameter) {

    outlinefile_filt <- Out(outlines_filt, fac = outlines.data_filt) # creation of an outline file with the database supplying metadata

    efourierfile <- efourier(outlinefile_filt, norm = TRUE)

    efourierfilepca <- PCA(efourierfile)
    str(efourierfilepca)

    ### distance and clustering

    # a distance matrix (default euclidean distance is correct)
    distobj <- dist(efourierfilepca$x[,1:10])
    summary(distobj)

    # ward hierclus
    cluswardd2 <- hclust(d = distobj, method = "ward.D2")

    ### silhouette

    # 'cutting' the dendrogramm (the 'tree') into k clusters
    clusloesk2 <- cutree(tree = cluswardd2, k=3)

    # silhouette
    # requires an integer vector holding cluster ids and a distance matrix
    sils <- silhouette(x = clusloesk2, dist=distobj)

    # overall performance is average silhouette width
    summary(sils)$avg.width

    ### "all together now"

    ## "for k in two to some" loop
    
    # empty vector for result
    silvec <- rep(0,9)

    for (k in 2:10)
    { clusloeskk <- cutree(tree = cluswardd2, k=k) # using the wardd2 dendrogr
    sils <- silhouette(x = clusloeskk, dist=distobj)
    silvec[k-1] <- summary(sils)$avg.width }
    (kopt <- (2:10)[which.max(silvec)]) # how many clusters are silhouette-optimal

    ## visualize optimal silhouette

    # by barplot
    png(paste0(dir,"R_statistics/GMM/GMM_silhoutte_bar.png"), width=8220, height=5086, res=1200)
    barplot(silvec, col=8, las=1, names.arg = 2:10, ylim=c(0,1), space=0, width=1)
    points(kopt-1.5, max(silvec)+0.1, pch=25, bg=2, cex=3)
    text(kopt-1.5, max(silvec)+0.18, round(max(silvec),2),font=3)
    
    dev.off()

    ## visualize clusters

    # in shape space scatter
    koptcol <- brewer.pal(n = 3, name = 'Dark2') # colorblind friendly
    koptcol <- adjustcolor(koptcol, 0.5) # transperent colors

    clusloeskopt  <- cutree(tree = cluswardd2, k=kopt) # using the wardd2 dendrogr

    png(paste0(dir,"R_statistics/GMM/GMM_scatter.png"), units="px", width=8220, height=5086, res=1200)
    plot(efourierfilepca , pch="", cex=2) # empty shape space

    par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE)
    points(efourierfilepca$x , pch = 21, bg = koptcol[clusloeskopt], cex = 1) # col points
    text(efourierfilepca$x, labels = outlines.data_filt$older_younger, cex= 0.5) # plot age
    
    # add legend to top right, outside plot region
    legend("topright", title = "Cluster", c("1","2","3"), cex = .8, inset = c(0.05, 0.05), pch = 21, col = 'black', pt.bg = koptcol, pt.cex = 1)
    
    dev.off()
    
    ## cluster plot
    
    outlines.data_filt$clust <- clusloeskopt
    print(outlines.data_filt$clust)

    outlinefile_filt <- Out(outlines_filt, fac = outlines.data_filt) # creation of an outline 

    efourierfile <- efourier(outlinefile_filt, norm = TRUE)
    
    print(efourierfile)

    ## export PCA values

    scores_pca <- as.data.frame(efourierfilepca$x)
    scores_pca <- cbind(outlines.data_filt, scores_pca)
    scores_pca$clust <- clusloeskopt

    outlines.data_filt$clust <- clusloeskopt

    write.csv(scores_pca, paste0(dir,'R_statistics/GMM/PCA-clust.csv',sep=''))
    
    write.csv(outlines.data_filt, paste0(data_metrics, '_GMM_clust.csv', sep=''))

    ##

    efourierfilepca$clust <- outlines.data_filt$clust
    png(paste0(dir,"R_statistics/GMM/GMM_scatter_1.png"), units="px", width=8220, height=5086, res=1200)
    plot(efourierfilepca, ordered (efourierfilepca$clust, levels=c(1,2,3)), xax = 1, yax = 2, points = T, cex = 0.5,center.origin = F, grid = TRUE, col = brewer.pal(n = 3, name = 'Dark2'), morphospace = T, ellipses = T, conf.ellipses = 0.95, chull = F, chull.filled = F, chull.filled.alpha = 0.92, eigen = FALSE, rug = FALSE, title = "", labelsgroups=TRUE, cex.labelsgroups=1.15, rect.labelsgroups = FALSE, color.legend = T, old.par = F, nb.grids = 1)
    par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE)
    
    dev.off()

    ## K-means

    png(paste0(dir,"R_statistics/GMM/GMM_kmeans.png"), units="px", width=8220, height=5086, res=1200)
    KMEANS(efourierfilepca, centers = 3)
    par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE)
    
    dev.off()
    
    png(paste0(dir,"R_statistics/GMM/GMM_meanshapes_",parameter,".png"), units="px", width=8220, height=5086, res=1200)    
    c <- MSHAPES(efourierfile, parameter, nb.pts=1200)$shp
    plot_MSHAPES(c)
    par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE)
    
    dev.off()
                
    return (efourierfilepca)
}

GMM_function <- function(dir, data_metrics, parameter) {

    dir.create(file.path(dir, "R_statistics/GMM/"), showWarnings = FALSE)

    outlines.data <- read.csv(paste0(data_metrics, '.csv'), header = T, sep=",", encoding = "UTF-8")
    GMM_prep_return <- GMM_filt(dir, outlines.data)

    outlines_filt <- GMM_prep_return$outlines_filt
    outlines.data <- GMM_prep_return$outlines.data

    outlines.data$X <- NULL

    data <- GMM(dir, outlines_filt, outlines.data, data_metrics, parameter)

    return (outlines.data)
}