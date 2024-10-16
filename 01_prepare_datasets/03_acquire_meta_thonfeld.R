##### data from Thonfeld et al. (2022)
### https://download.geoservice.dlr.de/TCCL/files/tccl_README.txt
### this script extracts sample points from the Thonfeld et al. (2022) dataset
### which are later used to extract Sentinel-2 time series for those pixels/points
### from a Sentinel-2 datacube (see following scripts)
### these data are used in pre-training as disturbance data


# Specify your packages
my_packages <- c("terra", "raster", "foreach", "stars", "parallel", 
                 "stringr", "rgdal", "foreach", "lubdridate", "dplyr", "exactextractr")
# Extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , "Package"])]
# Install not installed packages
if(length(not_installed)) install.packages(not_installed)


set.seed(1)

### imports
require(raster)
require(rgdal)
require(dplyr)
require(stars)
library(lubridate)
require(terra)
require(tictoc)
require(foreach)
require(exactextractr)
require(rgeos)
require(parallel)

# set working directory
setwd('/home/cangaroo/christopher/future_forest/forest_decline/data')


# load raster data
rast <- raster("path/to/raster.tif")

### filter raster for values from 1-40 (remove 0 and 100 (= set NA))
# (0 = non-disturbance, 100 = non-forested)

### use cluster to speed up the process
beginCluster(n = 70)
rast <- clusterR(rast, clamp, args=list(lower=1, upper=40, useValues=FALSE))
endCluster()

# ### save raster quickly to save processing time when repeating
writeRaster(rast, filename = "path/to/raster_clamped.tif", format = "GTiff", overwrite = TRUE)


################## if processed already, step in here: ###################
rast <- raster("path/to/raster_clamped.tif")

######## prepare further processing of e.g. non-disturbance layers (necessary for other scripts)
rast <- rast("1_raw_datasets/thonfeld_data/thonfeld_clamped.tif")

dir.create("path/to//tiles")
dir.create("path/to/temp_sample_polys")

### subset into tiles
r <- rast(ncols = 25, nrows = 25)
values(r) <- 1:ncell(r)
crs(r) <- crs(rast)
ext(r) <- ext(rast)
ff <- makeTiles(rast, r,
                filename = 'path/to/tiles/thonfeld_tile_.tif', na.rm = TRUE,
                overwrite = TRUE)
vrt(ff, filename = 'path/to/thonfeld_all_tiles.vrt',
    overwrite = TRUE)
#### end of vrt file creation

### prepare looping through tiles for sample generation
list_tiles <- list.files('path/to/thonfeld_data/tiles', pattern = ".tif$", 
                         full.names = TRUE)


### start foreach loop here: iterate over tiles
no_cpus <- parallel::detectCores() - 3

my.cluster <- parallel::makeCluster(
  no_cpus,
  type = "FORK"
)

#register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)

foreach (par_iterator = 1:length(list_tiles)) %dopar% {

  rast <- rast(list_tiles[par_iterator])

  ### mask out senf data (assuming that senf script has been run before)
  dat <- rast("path/to/senf.vrt")
  # make sure to exclude all the possible intersecting datasets (in time and space)
  dat <- crop(dat, ext(rast))
  dat <- sf::as_Spatial(sf::st_as_sf(stars::st_as_stars(dat), 
                                             as_points = FALSE, merge = TRUE))
  dat <- rgeos::gBuffer(dat, byid = TRUE, width = 0)
  dat <- dat[dat$germany_all_tiles > 2015, ]
  dat <- st_as_sf(dat)
  dat <- st_buffer(st_geometry(dat), 30) # 30m buffer
  rast <- terra::mask(rast, vect(dat), inverse = TRUE)
  
  # forwind storm dataset
  dat <- vect("path/to/forwind", crs = 'EPSG:3035')
  dat <- crop(dat, ext(rast))
  if (nrow(dat) > 0){
    dat <- dat[as.Date(dat$date) > as.Date('2015-01-01'), ]
    dat <- st_as_sf(dat)
    dat <- st_buffer(st_geometry(dat), 30) # 30m buffer
    rast <- terra::mask(rast, vect(dat), inverse = TRUE)
  }
  rm(dat)
  
  # convert to polygons
  poly <- sf::as_Spatial(sf::st_as_sf(stars::st_as_stars(rast), 
                                           as_points = FALSE, merge = TRUE)
  ) # requires the sf, sp, raster and stars packages
  
  # repair polygons
  poly <- rgeos::gBuffer(poly, byid = TRUE, width = 0)
    
  ### get sample points
  ### dissolve all polygons of tile
  poly <- vect(poly)
  poly <- terra::aggregate(poly, count = FALSE)
  
  ### get number of points defined by point density
  no_pts <- ifelse(as.integer(round(sum(expanse(poly)) / 25000, 0)) > 0,
                   as.integer(round(sum(expanse(poly)) / 25000, 0)), 1)

  ### convert to sf object
  poly <- st_as_sf(poly)
  ### sample points
  sample_points <- st_sample(poly, no_pts)
  
  # remove points too close to each other
  points_matrix <- gWithinDistance(as(sample_points, 'Spatial'), dist = 50, byid = TRUE)
  points_matrix[lower.tri(points_matrix, diag=TRUE)] <- NA
  v <- colSums(points_matrix, na.rm=TRUE) == 0
  sample_points <- sample_points[v]
  
  ### if no sample points in this tile (because of no forest, for instance), skip
  if (length(sample_points) > 0) {
    
    ### convert to SpatVector
    sample_points <- vect(sample_points)
    
    ### get those pixels which intersect with sample_points
    rast <- terra::mask(rast, sample_points, inverse = FALSE, touches = TRUE)
    
    ### convert those pixels to polygons
    sample_polys <- as.polygons(rast, dissolve = FALSE, values = TRUE, 
                                na.rm = TRUE, trunc = FALSE)   
    
    ### intersect with copernicus forest layer for meta information
    ### get forest type information
    forest <- rast('1_raw_datasets/copernicus_forest_layers/germany/DATA/germany.vrt')
    forest <- crop(forest, ext(sample_polys))
    forest[forest == 0, ] <- NA
    forest[forest == 1, ] <- 0
    forest[forest == 2, ] <- 1
    forest[forest > 2, ] <- NA

    ### note that 1 = coniferous, 0 = broadleaved
    df <- exact_extract(forest, st_as_sf(sample_polys), progress = FALSE, fun = 'mean')
    ### add information
    sample_polys$frac_coniferous <- df
  
    ### add date column
    # create random date between disturbance year + 12 months and disturbance year + 3 years for each polygon
    # -> all polygons show trees that died either between 2018-01-01 and 2021-04-30
    # -> a 4-year period has to end by 2021-01-01 so that every dieback process is covered
    sample_polys$dist_month <- as.integer(sample_polys$layer)
    sample_polys$layer <- as.integer(sample_polys$layer)
    sample_polys$layer <- as.character(as.Date('2018-01-01') %m+% months(sample_polys$layer))
    names(sample_polys)[1] <- 'date'
    
    # preserve disturbance year
    sample_polys$year <- substr(sample_polys$date, 1, 4)
    sample_polys <- as(sample_polys, 'Spatial')
    
    # get random enddate up to 6 months later than actual enddate
    sample_polys@data <- sample_polys@data %>% rowwise() %>%  
      mutate(date = sample(x = seq(from = as.Date(date, format='%Y-%m-%d'), 
                                    to = as.Date(date, format='%Y-%m-%d') %m+% months(6),
                                    by = "day"), 
                           size = 1))

    sample_polys@data <- as.data.frame(sample_polys@data)
      
    ### quantify disturbance
    ### it might make sense to use 'unknown' category, 
    ### since the paper cannot differentiate between disturbances
    ### -> we do not know, so it could be anything
    sample_polys@data$mort_0 <- 0.0 # no disturbance
    sample_polys@data$mort_1 <- 0.0 # harvest, thinning
    sample_polys@data$mort_2 <- 0.0 # biotic, e.g. bark beetle
    sample_polys@data$mort_3 <- 0.0 # abiotic disturbance: drought, lightning
    sample_polys@data$mort_4 <- 0.0 # gravitational event, uprooting, storm, windthrow
    sample_polys@data$mort_5 <- 1.0 # unknown
    sample_polys@data$mort_6 <- 0.0 # fire
    sample_polys@data$mort_7 <- 0.0
    sample_polys@data$mort_8 <- 0.0
    sample_polys@data$mort_9 <- 0.0
    
    ### add plotID
    sample_polys@data$plotID <- paste0(
      substr(sapply(strsplit(list_tiles[par_iterator], "/"), tail, 1), 
             1, nchar(sapply(strsplit(list_tiles[par_iterator], "/"), tail, 1))-4), 
      '_', 1:length(sample_polys))
    
    ### get country and region information
    countries <- vect("/path/to/NUTS_RG_01M_2021_3035.shp")
    countries <- countries[countries$LEVL_CODE == 1, ]
    countries <- st_as_sf(countries)
    countries <- countries[, c(4)]
    sample_polys <- sp_as_sf(sample_polys)
    sample_polys <- st_join(sample_polys, countries, largest = TRUE)
    sample_polys <- as(sample_polys, 'Spatial')
    
    # reorder column names
    sample_polys@data <- sample_polys@data %>%
      select(plotID, date, mort_0, mort_1, mort_2, mort_3, mort_4, 
             mort_5, mort_6, mort_7, mort_8, mort_9, everything())

    ### write to disk
    writeOGR(obj=sample_polys, dsn=paste0("path/to/temp_sample_polys/temp_sample_polys_", par_iterator), 
             layer=paste0("temp_sample_polys_", par_iterator, ".gpkg"), driver="GPKG")
  }
  
### end of large foreach loop
}

parallel::stopCluster(cl = my.cluster)

### put together the data
### merge/bind all polygons (created as sampling areas from the different tiles)
lst <- list.files("path/to/temp_sample_polys/", full.names = TRUE)
 
### bind together
bound <- do.call(rbind, lapply(lst, vect))
bound

### assign unique plotID per sample
bound$plotID <- paste0('thonfeld_', 1:nrow(bound), '_', gsub('-', '_', bound$date))

### improve destination folder and write to disk
writeVector(bound, "2_shapes/thonfeld_data_samples.gpkg")