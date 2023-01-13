### TITLE: PyREnArA (MEM) ###

### AUTHORS: Robin John (Institute of Prehistoric Archeology, University of Cologne, Bernhard-Feilchenfeld-Str. 11, 50969 Köln, Germany) 
### and Dr. Georg Roth (Institut für Prähistorische Archäologie, Freie Universität Berlin, Fabeckstr. 23-25, 14195 Berlin, Germany)



#install.packages(c(...)

# load required packages
require(ade4)
require(adespatial)
require(adegraphics)
require(spdep)
require(maptools)
require(raster)
require(geosphere)
require(geodist)



MEM_function <- function(directory, directory_to_data) {
    
    dir.create(file.path(directory, "R_statistics/MEM/"), showWarnings = FALSE)
    
    dataframe <- read.csv(directory_to_data, header = T, encoding = "UTF-8")

    coo <- rbind(dataframe$long, dataframe$lat)
    coo <- as.data.frame(t(coo))
    colnames(coo) <- c('long', 'lat')
    
    png(file=paste0(directory, "R_statistics/MEM/",'localities.png'), width=8333, height=5156, res=1200)
    
    plot(coo, xlim=c(15,19), ylim=c(47,51), pch=20, cex=2, col='red', xlab='X', ylab='Y', las=1)
    
    dev.off()
    
    mem.db <- dbmem(coo, MEM.autocor = c("all"), store.listw = TRUE)
    mem.db
    
    s.value(coo, mem.db, symbol = "circle", ppoint.cex = 0.8, xlim = c(14, 19), ylim = c(46, 51))
    barplot(attr(mem.db, "values"), main = "Eigenvalues of the spatial weighting matrix", cex.main = 0.7)

    listw.db <- attr(mem.db, "listw")
    dlist <- listw.db$weights
    listw.db <- nb2listw(listw.db$neighbours, glist = dlist)
    listw.db

    moranI <- moran.randtest(mem.db, listw.db)
    head(attr(mem.db, "values") / moranI$obs)

    s.label(coo, nb = listw.db)

    MC.env <- moran.randtest(dataframe$width, listw.db, nrepet = 999)
    MC.env

    mc.bounds <- moran.bounds(listw.db)
    mc.bounds

    metrics <- rbind(dataframe$width, dataframe$length, dataframe$l_w_index, dataframe$area, dataframe$percent_area, dataframe$contour_length, dataframe$MP_CM_x_offset, dataframe$MP_CM_y_offset)
    metrics <- as.data.frame(t(metrics))
    colnames(metrics) <- c('width', 'length', 'l_w_index', 'area', 'percent_area', 'contour_length', 'MP_CM_X_offset', 'MP_CM_y_offset')

    pca.hell <- dudi.pca(metrics, scale = TRUE, scannf = FALSE, nf = 2)

    moran.randtest(pca.hell$li, listw = listw.db)

    s.value(coo, pca.hell$li, symbol = "circle", col = c("orangered4", "palegreen4"), ppoint.cex = 0.8, xlim = c(14, 19), ylim = c(46, 51))

    ms.hell <- multispati(pca.hell, listw = listw.db, scannf = FALSE)
    summary(ms.hell)

    g.ms.maps <- s.value(coo, ms.hell$li, method = c("color"), symbol = "circle", col = c("orangered4", "orangered", "orange", "skyblue", "skyblue2", "skyblue4"), ppoint.cex = 0.6, xlim = c(14, 19), ylim = c(46, 51))

    g.ms.spe <- s.arrow(ms.hell$c1, plot = FALSE)
    g.ms.spe

    mem.dist.sel <- mem.select(pca.hell$tab, listw = listw.db)
    mem.dist.sel$global.test

    mem.dist.sel$summary
    
}
