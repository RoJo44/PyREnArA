### TITLE: PyREnArA (MEM) ###

### AUTHORS: Robin John (Institute of Prehistoric Archeology, University of Cologne, Bernhard-Feilchenfeld-Str. 11, 50969 Köln, Germany) 
### and Dr. Georg Roth (Institut für Prähistorische Archäologie, Freie Universität Berlin, Fabeckstr. 23-25, 14195 Berlin, Germany)




# load required packages
require(ade4)
require(adespatial)
require(adegraphics)
require(spdep)
require(maptools)
require(raster)
require(geosphere)
require(geodist)
require(vegan)

# cite ade4 package
print("Thioulouse, J., Dray, S., Dufour, A., Siberchicot, A., Jombart, T., Pavoine, S. (2018). Multivariate Analysis of Ecological Data with ade4. doi:10.1007/978-1-4939-8850-1.")

MEM_function <- function(directory, directory_to_data) {
    
    dir.create(file.path(directory, "R_statistics/MEM/"), showWarnings = FALSE)

    datfram <- read.csv(paste0(directory_to_data, '.csv'), header = TRUE, encoding = "UTF-8", sep=",")

    coo <- rbind(datfram$longitude, datfram$latitude)
    coo <- as.data.frame(t(coo))
    colnames(coo) <- c('longitude', 'latitude')

    # plot sites
    png(file=paste0(directory, "R_statistics/MEM/",'localities.png'), width=8333, height=5156, res=1200)
    plot(coo, xlim=c(-10, 2), ylim=c(35, 45), pch=20, cex=1.5, col='red', xlab='X', ylab='Y', las=1)
    dev.off()

    # distance based mem
    mem.db <- dbmem(coo, MEM.autocor = c("non-null"), store.listw = TRUE)

    # Print the (n-1) non-null eigenvalues
    attributes(mem.db)$values

    # plot spatial weightening matrices
    png(file=paste0(directory, "R_statistics/MEM/",'swm_distance_based.png'), width=8333, height=5156, res=1200)
    s.value(coo, mem.db[,1:4], symbol = "circle", ppoint.cex = 0.8, xlim=c(-10, 2), ylim=c(35, 45))
    dev.off()

    # barplot for spatial weightening matrices
    png(file=paste0(directory, "R_statistics/MEM/",'swm_barplot.png'), width=8333, height=5156, res=1200)
    barplot(attr(mem.db[,1:4], "values"), main = "Eigenvalues of the spatial weighting matrix", cex.main = 0.7)
    dev.off()

    listw.db <- attr(mem.db, "listw")
    dlist <- listw.db$weights
    listw.db <- nb2listw(listw.db$neighbours, glist = dlist)
    listw.db

    moranI <- moran.randtest(mem.db, listw.db)
    head(attr(mem.db, "values") / moranI$obs)

    mc.bounds <- moran.bounds(listw.db)
    mc.bounds

    # load metrics
    metrics <- rbind(datfram$width, datfram$length, datfram$l_w_index, datfram$area, datfram$percent_area, datfram$contour_length, datfram$MP_CM_x_offset, datfram$MP_CM_y_offset)
    metrics <- as.data.frame(t(metrics))
    colnames(metrics) <- c('width', 'length', 'l_w_index', 'area', 'percent_area', 'contour_length', 'MP_CM_X_offset', 'MP_CM_y_offset')

    # pca
    pca.hell <- dudi.pca(metrics, scale = TRUE, scannf = FALSE, nf = 2)

    # moran randtest
    moran_randtest <- moran.randtest(pca.hell$li, listw = listw.db)
    moran_randtest
    #write.csv(moran_randtest,paste0(dir,'R_statistics/GMM/moran_randtest.csv',sep=''))

    # plot pca
    png(file=paste0(directory, "R_statistics/MEM/",'morans_pca.png'), width=8333, height=5156, res=1200)
    s.value(coo, pca.hell$li, symbol = "circle", ppoint.cex = 1, xlim=c(-10, 2), ylim=c(35, 45))
    dev.off()

    ms.hell <- multispati(pca.hell, listw = listw.db, scannf = FALSE)
    summary(ms.hell)

    png(file=paste0(directory, "R_statistics/MEM/",'msmaps.png'), width=8333, height=5156, res=1200)
    g.ms.maps <- s.value(coo, ms.hell$li, symbol = "circle", ppoint.cex = 1, xlim=c(-10, 2), ylim=c(35, 45))
    dev.off()

    png(file=paste0(directory, "R_statistics/MEM/",'arrowmaps.png'), width=8333, height=5156, res=1200)
    g.ms.spe <- s.arrow(ms.hell$c1, plot = FALSE)
    g.ms.spe
    dev.off()

    mem.dist.sel <- mem.select(pca.hell$tab, listw = listw.db)
    mem.dist.sel$global.test

    mem.dist.sel$summary
}