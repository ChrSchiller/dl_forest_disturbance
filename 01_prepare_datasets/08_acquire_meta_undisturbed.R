##### note that this script has to be executed AFTER the creation of
##### disturbance data
##### cause it relies on their output

### this script uses the Copernicus Land Monitoring Service (forest mask)
### to sample undisturbed forest after removing known disturbances from other datasets
### output is a metafile with presumably undisturbed forest polygons
### (resembling Sentinel-2 pixels) which can be used for pre-training the model

set.seed(123)

# Specify your packages
my_packages <- c("terra", "raster", "sp", "prevR", "sf", "stars", 
                 "stringr", "rgdal", "foreach", "lubdridate", "dplyr", "exactextractr", 
                 "rgeoboundaries", "rgeos")
# Extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , "Package"])]
# Install not installed packages
if(length(not_installed)) install.packages(not_installed)

require(terra)
require(raster)
require(sp)
require(prevR)
require(sf)
require(stars)
library(stringr)
require(rgdal)
require(foreach)
library(lubridate)
require(dplyr)
require(exactextractr)
require(rgeoboundaries)
require(rgeos)

### function that shrinks polygons by 'size' meters if possible; if not, does not do anything
### adapted from:
# https://gis.stackexchange.com/questions/392505/can-i-use-r-to-do-a-buffer-inside-polygons-shrink-polygons-negative-buffer?rq=1
# negative buffer size is given as a positive value
shrinkIfPossible <- function(sf, size) {
  # compute inward buffer
  sg <- st_buffer(st_geometry(sf), -size)
  
  # update geometry only if polygon is not degenerate
  st_geometry(sf)[!st_is_empty(sg)] = sg[!st_is_empty(sg)]
  
  # return updated dataset
  return(sf)
}



### setwd
workdir <- ''
setwd(workdir)

datasets <- c('germany') 

### loop through countries (here: only Germany)
for (i in 1:length(datasets)){

  dir.create(paste0('1_raw_datasets/copernicus_forest_layers/', datasets[i], '/DATA_smaller_tiles'))

  # forest cover
  lst <- list.files(paste0('1_raw_datasets/copernicus_forest_layers/', datasets[i], '/DATA'),
                    pattern = '\\.tif$', full.names = TRUE)

  no_cpus <- min(parallel::detectCores() - 3, length(lst))

  my.cluster <- parallel::makeCluster(
    no_cpus,
    type = "FORK"
  )

  #register it to be used by %dopar%
  doParallel::registerDoParallel(cl = my.cluster)

  foreach (iter = 1:length(lst)) %dopar% {

    rast <- rast(lst[iter])
    r <- rast(ncols = 2, nrows = 2)
    values(r) <- 1:ncell(r)
    crs(r) <- crs(rast)
    ext(r) <- ext(rast)

    ### create smaller tiles to facilitate processing
    terra::makeTiles(rast, r, filename = paste0('1_raw_datasets/copernicus_forest_layers/',
                                               datasets[i], '/DATA_smaller_tiles/',
                                         datasets[i], '_',
                                         substr(sapply(
                                           strsplit(lst[iter], "/"), tail, 1),
                                           1, nchar(sapply(strsplit(lst[iter], "/"),
                                                           tail, 1))-4), '_.tif'),
                     na.rm = TRUE, overwrite = TRUE)
    

  }

  parallel::stopCluster(cl = my.cluster)

  # combine as vrt for later use
  vrt(list.files(paste0('1_raw_datasets/copernicus_forest_layers/',
                        datasets[i], '/DATA_smaller_tiles'), pattern = '\\.tif$', full.names = TRUE),
      filename = paste0('1_raw_datasets/copernicus_forest_layers/', datasets[i],
      '/DATA_smaller_tiles/', datasets[i], '.vrt'), overwrite = TRUE)

}


### put together a vector with all tiles
list_fls <- c()

for (i in 1:length(datasets)){

  list_fls <- c(list_fls, list.files(paste0('1_raw_datasets/copernicus_forest_layers/', datasets[i], '/DATA_smaller_tiles'),
                                     pattern = '\\.tif$', full.names = TRUE))

}

### register another cluster
no_cpus <- 40

my.cluster <- parallel::makeCluster(
  no_cpus,
  type = "FORK"
)

#register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)

foreach(par_iterator = 1:length(list_fls)) %dopar% {

  cntry_string <- str_extract(list_fls[par_iterator], str_c(datasets, collapse = "|"))

  rast <- rast(list_fls[par_iterator])

  cntry <- geoboundaries(country = cntry_string, type = 'HPSCGS')
  cntry <- vect(cntry)
  cntry <- terra::project(cntry, 'epsg:3035')
  
  ### load disturbance data to be removed
  # senf dataset
  senf2 <- rast(paste0(getwd(),
                       '/1_raw_datasets/senf2/raw_data/',
                       cntry_string, '/', cntry_string, '_all_tiles_year.vrt'))

  # forwind storm dataset
  forwind <- vect(paste0(getwd(), '/2_shapes/forwind_plots_epsg3035_extent'), crs = 'EPSG:3035')

  # thonfeld dataset
  if (cntry_string == 'germany'){
    thonfeld <- rast(paste0(getwd(), '/1_raw_datasets/thonfeld_data/thonfeld_all_tiles.vrt'))
  }

  rast <- terra::mask(rast, cntry, inverse = FALSE)
  
  ### mask by senf data
  senf2 <- terra::crop(senf2, ext(rast))
  resmask <- resample(senf2, rast)
  rast <- terra::mask(rast, resmask, maskvalues = c(1986:2020))
  rm(senf2)
  
  ### mask by forwind if necessary
  if (relate(ext(forwind), ext(rast), "intersects")){
    rast <- terra::mask(rast, forwind, inverse = TRUE)
    rm(forwind)
  }
  
  ### mask by thonfeld if necessary
  if (relate(ext(thonfeld), ext(rast), "intersects")){
    thonfeld <- terra::crop(thonfeld, ext(rast))
    resmask <- resample(thonfeld, rast)
    rast <- terra::mask(rast, resmask, maskvalues = c(1:40))
    rm(resmask)
    rm(thonfeld)
  }

  ### remove the finetuning datasets (Schwarz, Schiefer, FNEWS, AOIs) in the same manner

  ### write temp file to disk
  terra::writeRaster(rast,
                     paste0('1_raw_datasets/copernicus_forest_layers/temp/temp_all_tiles_all_cntrys/',
                            substr(
                              sapply(strsplit(list_fls[par_iterator], "/"), tail, 1),
                              1, nchar(sapply(strsplit(list_fls[par_iterator], "/"), tail, 1))-4),
                            '.tif'), overwrite = TRUE)
  rm(rast)
}

parallel::stopCluster(cl = my.cluster)

### foreach loop for all the tiles
lst <- list.files('1_raw_datasets/copernicus_forest_layers/temp/temp_all_tiles_all_cntrys',
                  pattern = '.tif$', full.names = TRUE)

### loop through all the tiles in parallel
no_cpus <- 50

# detach(package:snowfall)
my.cluster <- parallel::makeCluster(
  no_cpus,
  type = "FORK", 
  outfile=""
)

#register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)

# check if it is registered (optional)
foreach::getDoParRegistered()
my.cluster

foreach (i = 1:length(lst)) %dopar% {

  ### read the tile
  rast <- rast(lst[i])
  
  ### some rasters do not contain any values although hasValues(rast) == TRUE
  ### filter them out beforehand
  if (sum(values(rast), na.rm = TRUE) > 0){
    poly <- sf::as_Spatial(sf::st_as_sf(stars::st_as_stars(rast),
                                             as_points = FALSE, merge = TRUE)
    ) 
    poly <- rgeos::gBuffer(poly, byid = TRUE, width = 0)
    rm(rast)
  
    ### shrink if possible to make sure sampled pixels are completely within the forest
    # get size column
    poly$size <- area(poly)
  
    poly_shrinked <- st_as_sf(poly) 
    poly_shrinked <- shrinkIfPossible(poly_shrinked, 30)
    poly_shrinked <- as(poly_shrinked, 'Spatial')
    poly_shrinked$size_shrinked <- area(poly_shrinked)
    poly_subset <- poly_shrinked[!(poly_shrinked@data$size == poly_shrinked@data$size_shrinked) , ]
  
    ### remove all polygons with size < 100 m2 (which is smaller than a S2 pixel)
    poly_subset <- poly_subset[poly_subset$size_shrinked > 100, ]
    rm(poly_shrinked)
  
    ### add another constraint: sometimes, shrinking and filtering
    ### leads to empty objects
    if (nrow(poly_subset) > 0){
  
      ### dissolve all polygons of tile
      poly_subset <- vect(poly_subset)
      poly <- terra::aggregate(poly_subset, count = FALSE)
      rm(poly_subset)
    
      ### get number of points defined by point density
      no_pts <- ifelse(as.integer(round(sum(expanse(poly)) / 500000, 0)) > 0,
                       as.integer(round(sum(expanse(poly)) / 500000, 0)), 1)
      poly <- st_as_sf(poly)
      sample_points <- st_sample(poly, no_pts)
    
      points_matrix <- gWithinDistance(as(sample_points, 'Spatial'), dist = 50, byid = TRUE) # enforce min distance
      points_matrix[lower.tri(points_matrix, diag=TRUE)] <- NA

      v <- colSums(points_matrix, na.rm=TRUE) == 0
      sample_points <- sample_points[v]
    
      ### build polygons (10x10m pixels)
      radius <- 5
    
      coords <- st_coordinates(sample_points)
    
      # define the plot edges based upon the plot radius.
      yPlus <- coords[, 2] + radius
      xPlus <- coords[, 1] + radius
      yMinus <- coords[, 2] - radius
      xMinus <- coords[, 1] - radius
    
      # calculate polygon coordinates for each plot centroid.
      square=cbind(xMinus, yPlus,  # NW corner
                   xPlus, yPlus,  # NE corner
                   xPlus, yMinus,  # SE corner
                   xMinus, yMinus, # SW corner
                   xMinus, yPlus)  # NW corner again - close ploygon
    
      # Extract the plot ID information
      ID <- 1:length(sample_points)
      rm(sample_points)
    
      ### assign preliminary plotid (e.g. country name)
      runnum <- as.factor(paste0(
        substr(sapply(strsplit(lst[i], "/"), tail, 1), 1, nchar(sapply(strsplit(lst[i], "/"), tail, 1))-4), '_', ID)
        )
    
      # create spatial polygons from coordinates
      polys <- SpatialPolygons(mapply(function(poly, id) {
        xy <- matrix(poly, ncol=2, byrow=TRUE)
        Polygons(list(Polygon(xy)), ID=id)
      },
      split(square, row(square)), runnum),
      proj4string=CRS('+init=epsg:3035'))

      # Create SpatialPolygonDataFrame -- this step is required to output multiple polygons
      plots_geo <- SpatialPolygonsDataFrame(polys, data.frame(id=runnum, row.names=runnum))
      rm(polys)
      colnames(plots_geo@data) <- c('plotID')

      ### get forest type information
      forest <- rast(paste0('1_raw_datasets/copernicus_forest_layers/',
                          sub("\\_.*", "", sapply(strsplit(lst[i], "/"), tail, 1)), '/DATA/',
                          sub("\\_.*", "", sapply(strsplit(lst[i], "/"), tail, 1)), '.vrt'))
      forest <- crop(forest, ext(plots_geo))
      forest[forest == 0, ] <- NA
      forest[forest == 1, ] <- 0
      forest[forest == 2, ] <- 1
      forest[forest > 2, ] <- NA
    
      ### note that 1 = coniferous, 0 = broadleaved
      df <- exact_extract(forest, plots_geo, progress = FALSE, fun = 'mean')
      ### assign value
      plots_geo@data$frac_coniferous <- df
    
      ### assign random date: the disturbance information reaches until May 2021
      ### (Thonfeld dataset is most up to date)
      ### + we aim at Sentinel-2-only data (which began around July 2015)
      ### therefore, random dates should be between 2019-07-01 and 2021-05-01
      # initialize field
      plots_geo@data$date <- as.Date('2019-07-01', format='%Y-%m-%d')
      plots_geo@data <- plots_geo@data %>% rowwise() %>%  mutate(date = sample(x = seq(from = as.Date('2019-07-01', format='%Y-%m-%d'),
                                                                                         to = as.Date('2021-05-01', format='%Y-%m-%d'),
                                                                                         by = "day"),
                                                                                 size = 1))
      plots_geo@data <- as.data.frame(plots_geo@data)
    
      ### quantify disturbance
      plots_geo$mort_0 <- 1.0 # no disturbance
      plots_geo$mort_1 <- 0.0 # harvest, thinning
      plots_geo$mort_2 <- 0.0 # biotic, e.g. bark beetle
      plots_geo$mort_3 <- 0.0 # abiotic disturbance: drought, lightning
      plots_geo$mort_4 <- 0.0 # gravitational event, uprooting, storm, windthrow
      plots_geo$mort_5 <- 0.0 # unknown
      plots_geo$mort_6 <- 0.0 # fire
      plots_geo$mort_7 <- 0.0
      plots_geo$mort_8 <- 0.0
      plots_geo$mort_9 <- 0.0
    
      # reorder column names
      plots_geo@data <- plots_geo@data %>%
        select(plotID, date, mort_0, mort_1, mort_2, mort_3, mort_4,
               mort_5, mort_6, mort_7, mort_8, mort_9, everything())
      
      plots_geo@data$cntry <- str_extract(lst[i], str_c(datasets, collapse = "|"))
      
      ### get country and region information
      countries <- vect("path/to/NUTS_RG_01M_2021_3035.shp")
      countries <- countries[countries$LEVL_CODE == 1, ]
      countries <- st_as_sf(countries)
      countries <- countries[, c(4)]
      plots_geo <- sp_as_sf(plots_geo)
      plots_geo <- st_join(plots_geo, countries, largest = TRUE)
      plots_geo <- as(plots_geo, 'Spatial')
      
      writeOGR(obj=plots_geo,
               dsn=
                 paste0('1_raw_datasets/copernicus_forest_layers/temp/temp_all_tiles_all_cntrys/temp_sample_polys_per_tile/',
                        substr(sapply(strsplit(lst[i], "/"), tail, 1),
                               1, nchar(sapply(strsplit(lst[i], "/"), tail, 1))-4)),
               layer=paste0(substr(sapply(strsplit(lst[i], "/"), tail, 1),
                                   1, nchar(sapply(strsplit(lst[i], "/"), tail, 1))-4), '.gpkg'), driver='GPKG')
    }
  }
### end of foreach loop
}

parallel::stopCluster(cl = my.cluster)

### put together the data
### merge/bind all polygons (created as sampling areas from the different tiles)
lst <- list.files('1_raw_datasets/copernicus_forest_layers/temp/temp_all_tiles_all_cntrys/temp_sample_polys_per_tile'
                  ,full.names = TRUE, include.dirs = TRUE)

### bind together the files
bound <- do.call(rbind, lapply(lst, vect))
bound

### assign unique plotID per sample
bound$plotID <- paste0('undstrbd_', bound$cntry, '_', 1:nrow(bound), '_', gsub('-', '_', bound$date))

### write to disk
writeVector(bound, "2_shapes/undstrbd_data_samples.gpkg", overwrite = TRUE)