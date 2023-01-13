### TITLE: PyREnArA (RDA) ###

### AUTHORS: Florian Linsel (Institute of Computer Science, Martin-Luther-Universität Halle-Wittenberg, Von-Seckendorff-Platz 1, 06120 Halle) 
### and Dr. Georg Roth (Institut für Prähistorische Archäologie, Freie Universität Berlin, Fabeckstr. 23-25, 14195 Berlin, Germany)



# load required packages
require(vegan)

# cite vegan
print("Oksanen J, Simpson G, Blanchet F, Kindt R, Legendre P, Minchin P, O'Hara R, Solymos P, Stevens M, Szoecs E, Wagner H, Barbour M, Bedward M, Bolker B, Borcard D, Carvalho G, Chirico M, De Caceres M, Durand S, Evangelista H, FitzJohn R, Friendly M, Furneaux B, Hannigan G, Hill M, Lahti L, McGlinn D, Ouellette M, Ribeiro Cunha E, Smith T, Stier A, Ter Braak C, Weedon J (2022). vegan: Community Ecology Package. R package version 2.6-4, URL https://CRAN.R-project.org/package=vegan ")



RDA_function <- function(directory, directory_to_data, parameter) {
    
    # create folder for data 
    dir.create(file.path(directory, "R_statistics/RDA/"), showWarnings = FALSE)
    
    ### calculation ###
    
    # load data
    datfram <- read.csv(paste0(directory_to_data, '.csv'), header=TRUE, sep=",", dec=".", row.names = 'ID')
    
    # seperate depending variables from explanatory variables
    Y <- datfram[,c('width', 'length', 'l_w_index', 'area', 'percent_area', 'contour_length', 'MP_CM_x_offset', 'MP_CM_y_offset', 'FD')]

    x <- datfram[,parameter]
    
    # run vegan´s RDA
    res <- vegan::rda(Y~x, scale=TRUE)

    # check if RDA has detected structure
    # if the test is not significant,
    # rethink the assumption of cause and effect relationship
    # "*" indicates significant results
    anova_cca <- vegan::anova.cca(res, alpha=0.05, beta=0.001, perm.max=9999)
    print(anova_cca)
    write.csv(anova_cca,paste0(directory, "R_statistics/RDA/", 'anova_cca.csv'))
    
    # calculate the explained portion
    explained_portion <- vegan::RsquareAdj(res)[[2]]
    
    # portions of the scatter per axis
    rda_summary <- summary(res)$cont 
    write.csv(rda_summary,paste0(directory, "R_statistics/RDA/", 'rda_summary.csv'))
    summary(res)$cont$importance[2,1]
    print(paste('explained proportion: ', explained_portion))

    ### presentation ###

    ### Screeplot
    (eigvalperc <- summary(res)$cont$importance[2,])

    (ymax <- round(100*max(eigvalperc)+5,0))
    
    png(file=paste0(directory, "R_statistics/RDA/", 'RDA_Eigenvalue.png'), width=8333, height=5156, res=1200)

    barplot(100*eigvalperc, las=2, names.arg=names(eigvalperc), cex.names=.7,
          space=0, width=1, ylim=c(0,ymax ),
          main="Eigenwertdiagramm",
          ylab="Prozent erkl?rter Streuung pro Achse")

    dev.off()

    ### Triplot
    k <- length(eigvalperc) # number of axis

    # coordinates of cases
    fallkoo <- scores(res, choices = c(1:k), display = c("sp","wa","bp"), scaling = 1, 1)$sites
    
    # coordinates of traits
    mmalkoo <- scores(res, choices = c(1:k), display = c("sp","wa","bp"), scaling = 1, 1)$species

    # coordinates of explanatory variable
    kk <- length(res$CCA$eig) # number of canonical axes (here one)
    (covar <- scores(res, choices = c(1:k), display = c("sp","wa","bp"), scaling = "sites", 1)$biplot)
    
    # limits of diagram
    (lim <- 0.5)
    (lim <- c(-1*lim,lim))
    
    # plot
    png(file = paste0(directory, "R_statistics/RDA/", 'RDA_Triplot.png'), width=8333, height=5156, res=1200)

    plot(mmalkoo, type="n", asp=1, xlim=lim, ylim=lim, las=1, xlab="", ylab="")

    # grid
    abline(v=0, lty=3, col=gray(.7))
    abline(h=0, lty=3, col=gray(.7))

    # cases
    rcol <- rgb(.8,.8,.8,.3) # transparent gray
    rbcol<- gray(.2) # dark gray
    points(fallkoo[,1:2], pch=21, bg=rcol, col=rbcol, cex=0.6)	# row points

    # traits
    cbcol<- gray(.2)
    arrows(0,0,mmalkoo[,1],mmalkoo[,2], lwd=1, length=.05, angle=15, col=cbcol)

    tcol <- rgb(.2,.2,.2,.6) # text color
    text(mmalkoo, rownames(mmalkoo), pos=3, cex=.8, font=3, col=tcol) # annotation for cols

    # explanatory variable
    arrows(0,0,covar[,1],covar[,2], lwd=3, length=.1, angle=15, col=cbcol)
    text( covar[,1]*1.05, covar[,2]*1.05, parameter, pos=4, cex=1.2, font=4)

    # subtext
    proz <- round(100*eigvalperc[1:2],1)
    subtextri <- paste("Triplot shows", sum(proz), "% of the variance")

    title(main="Triplot of RDA (canonic variable is cause)", cex.axis=.8,
        xlab="RDA I", ylab="PCA I", sub=subtextri, cex.sub=.8)

    dev.off()

    ### scatter explained by cause for each trait
    (inrexp <- inertcomp(object = res, display = "species", proportional = TRUE))

    yvlim <- 10
    yv <- round(100*max(inrexp)+5, -1)

    # limits
    oldpar <- par()
    par(mar = c(5,4,4,1))

    # positive and negative correlation with explanatory variable
    (negpos <- ifelse(mmalkoo[,1]<0, -1,1))
    
    # rank traits by explanable power
    ranks <- order(negpos*inrexp[,1])
    
    # plot
    png(file=paste0(directory, "R_statistics/RDA/", 'RDA_per_trait.png'), width=8333, height=5156, res=1200)

    barplot((negpos*100*inrexp[,1])[ranks], names.arg=rownames(inrexp)[ranks], las=2, cex.names=.7, space=0, width=1, ylim=c(-yvlim,yvlim))

    title(main="Canonic variable explaines variance of traits", ylab="% explained variance")

    dev.off()
    dev.off()

    ### Export RDA data 
    sc_si <- scores(res, display="sites", choices=c(1,2), scaling=1)
    datfram$RDA <- sc_si[,1:2]
    write.csv(datfram, paste0(directory, "R_statistics/RDA/",'data_metrics_rda.csv'), fileEncoding ="UTF-8")

}