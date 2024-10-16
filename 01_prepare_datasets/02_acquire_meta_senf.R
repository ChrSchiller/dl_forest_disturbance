### data from Senf and Seidl (2021)
### accessed at https://zenodo.org/record/3924381
### on 2022-11-02


##### note: this script has to be exectued BEFORE the thonfeld data preparation

### this script gets as input the raster files mentioned above
### and outputs polygons including disturbance/date/etc information
### for a set of countries in Central Europe

# Specify your packages
my_packages <- c("terra", "raster", "foreach", "stars", 
                 "stringr", "rgdal", "foreach", "lubdridate", "dplyr", "exactextractr")
# Extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , "Package"])]
# Install not installed packages
if(length(not_installed)) install.packages(not_installed)

### random seed
set.seed(123)

### imports
require(raster)
require(rgdal)
require(dplyr)
require(stars)
library(lubridate)
require(terra)
require(foreach)
require(exactextractr)
require(rgeos)
require(stringr)


# set working directory
setwd('/home/cangaroo/christopher/future_forest/forest_decline/data')

### get country names (= folder names)
### here: only Germany
fls <- list.files("path/to/senf/raw_data/")


##### this step takes the original country rasters and 
##### splits them into smaller tiles + adds a vrt file on top
##### this is done for both the year and the disturbance rasters similarly

### parallel processing with as many CPUs as countries concerned
no_cpus <- length(fls)

### define cluster
my.cluster <- parallel::makeCluster(
  no_cpus,
  type = "FORK"
)

### register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)

foreach(i = 1:length(fls)) %dopar% {
  ### load raster data
  year <- rast(paste0("1_raw_datasets/senf/raw_data/", fls[i], "/disturbance_year_1986-2020_", fls[i], ".tif"))

  ### create directory for disturbance year tiles
  dir.create(paste0("1_raw_datasets/senf/raw_data/", fls[i], "/year_tiles"))

  ### initialize tile sizes
  r <- rast(ncols = 25, nrows = 25)
  values(r) <- 1:ncell(r)
  crs(r) <- crs(year)
  ext(r) <- ext(year)

  ### make tiles and vrt
  ff <- makeTiles(year, r, filename = paste0("path/to/senf/raw_data/", fls[i], "/year_tiles/", fls[i], '_tile_.tif'), na.rm = TRUE)
  vrt(ff, filename = paste0("path/to/senf/raw_data/", fls[i], "/", fls[i], '_all_tiles_year.vrt'))

### end of foreach loop
}

### stop the cluster
parallel::stopCluster(cl = my.cluster)

### same for disturbance severity: each country raster gets one CPU
no_cpus <- length(fls)

### define and register cluster
my.cluster <- parallel::makeCluster(
  no_cpus,
  type = "FORK"
)
doParallel::registerDoParallel(cl = my.cluster)

### loop through countries and split disturbance severity rasters into smaller tiles 
foreach(i = 1:length(fls)) %dopar% {

  ### load raster data
  sev <- rast(paste0("1_raw_datasets/senf/raw_data/", fls[i], "/disturbance_severity_1986-2020_", fls[i], ".tif"))

  ### create output directory
  dir.create(paste0("path/to/senf/raw_data/", fls[i], "/severity_tiles"))

  ### initialize raster
  r <- rast(ncols = 25, nrows = 25)
  values(r) <- 1:ncell(r)
  crs(r) <- crs(sev)
  ext(r) <- ext(sev)

  ### make tiles and vrt
  ff <- makeTiles(sev, r, filename = paste0("path/to/senf/raw_data/", fls[i], "/severity_tiles/", fls[i], '_tile_.tif'), na.rm = TRUE)
  vrt(ff, filename = paste0("path/to/senf/raw_data/", fls[i], "/", fls[i], '_all_tiles_severity.vrt'))

### end of foreach loop creating tiles for disturbance severity of each country
}

### stop cluster
parallel::stopCluster(cl = my.cluster)


##### this step acquires sample points excluding other overlapping datasets (i.e., their polygons)
##### by looping through all of the tiles and creating the points randomly with 
##### specified minimum distance (to avoid self-overlaps within the dataset)

### initialize list of tiles
list_tiles <- c()

for (i in 1:length(fls)){
  ### add all tiles of each country
  list_tiles <- c(list_tiles, list.files(paste0('path/to/senf/raw_data/', fls[i], '/year_tiles'), pattern = ".tif$",
                           full.names = TRUE))
  
}
  
### assign number of CPUs for parallel processing
no_cpus <- parallel::detectCores() - 3

### start cluster
my.cluster <- parallel::makeCluster(
  no_cpus,
  type = "FORK"
)
doParallel::registerDoParallel(cl = my.cluster)

### loop through all tiles, remove conflicting datasets, 
### sample random points with specific density
foreach (par_iterator = 1:length(list_tiles)) %dopar% {

  ### load raster
  rast <- rast(list_tiles[par_iterator])
  
  ### get country information 
  cntry_string <- str_extract(list_tiles[par_iterator], str_c(fls, collapse = "|"))
  
  ### remove/mask out other datasets
  ### -> sampling pixels are clear of other disturbances already 
  ### -> we do not create sample points that we remove again later on
  ### -> no spatiotemporal autocorrelation between thonfeld data and others
  ### done on tile basis, since it is faster than doing it later on in "clean-up" scrip
  
  # forwind storm dataset
  dat <- vect(paste0(getwd(), '/path/to/forwind'), crs = 'EPSG:3035')
  if (relate(ext(dat), ext(rast), "intersects")){
    dat <- crop(dat, ext(rast))
    if (nrow(dat) > 0){
      dat <- st_as_sf(dat)
      dat <- st_buffer(st_geometry(dat), 30)
      rast <- terra::mask(rast, vect(dat), inverse = TRUE)
    }
  }
  rm(dat)
  
  
  # convert to polys
  poly <- sf::as_Spatial(sf::st_as_sf(stars::st_as_stars(rast), 
                                      as_points = FALSE, merge = TRUE)
  ) # requires the sf, sp, raster and stars packages
  poly <- rgeos::gBuffer(poly, byid = TRUE, width = 0)
  
  ### get sample points
  ### dissolve all polygons of tile
  poly <- vect(poly)
  poly <- terra::aggregate(poly, count = FALSE)
  ### get number of points defined by point density
  no_pts <- ifelse(as.integer(round(sum(expanse(poly)) / 25000, 0)) > 0,
                   as.integer(round(sum(expanse(poly)) / 25000, 0)), 1)
  
  
  poly <- st_as_sf(poly)
  ### sample points 
  sample_points <- st_sample(poly, no_pts)
  ### enforce minimum distance
  points_matrix <- gWithinDistance(as(sample_points, 'Spatial'), dist = 100, byid = TRUE)
  points_matrix[lower.tri(points_matrix, diag=TRUE)] <- NA
  v <- colSums(points_matrix, na.rm=TRUE) == 0
  sample_points <- sample_points[v]
  
  ### if no points available (e.g. no forest in this tile), skip
  if (length(sample_points) > 0) { # !(skip_tile) & 
    
    sample_points <- vect(sample_points)
    
    ### get those pixels which intersect with sample_points
    rast <- terra::mask(rast, sample_points, inverse = FALSE, touches = TRUE)
    ### convert to polygons
    sample_polys <- as.polygons(rast, dissolve = FALSE, values = TRUE, 
                                na.rm = TRUE, trunc = FALSE)
    
    ### intersect with copernicus forest layer for meta information
    ### get forest type information
    forest <- rast(paste0('1_raw_datasets/copernicus_forest_layers/', 
                   cntry_string, '/DATA/', cntry_string, '.vrt'))
    forest <- crop(forest, ext(sample_polys))
    
    forest[forest == 0, ] <- NA
    forest[forest == 1, ] <- 0
    forest[forest == 2, ] <- 1
    forest[forest > 2, ] <- NA

    ### note that 1 = coniferous, 0 = broadleaved
    df <- exact_extract(forest, st_as_sf(sample_polys), progress = FALSE, fun = 'mean') 
    ### add information
    sample_polys$frac_coniferous <- df
    rm(forest)
    
    ### load severity vrt
    sev <- rast(paste0('1_raw_datasets/senf/raw_data/', fls[i], '/', fls[i], '_all_tiles_severity.vrt'))
    sev <- crop(sev, ext(sample_polys))
    
    df <- exact_extract(sev, st_as_sf(sample_polys), progress = FALSE, fun = 'mean')
    sample_polys$mort_5 <- df # unknown disturbance agent, as senf data is unreliable in this regard
    
    ### add date column
    names(sample_polys)[1] <- 'year'
    sample_polys$year <- as.character(sample_polys$year)
    sample_polys <- as(sample_polys, 'Spatial')
    ## convert year to date
    sample_polys$date <- as.Date(paste0(sample_polys$year, '-01-01'), format = '%Y-%m-%d') %m+% months(12)
    
    ### random enddate up to 6 months later
    sample_polys@data <- sample_polys@data %>% rowwise() %>%  
      mutate(date = sample(x = seq(from = as.Date(date, format='%Y-%m-%d'), 
                                   to = as.Date(date, format='%Y-%m-%d') %m+% months(6),
                                   by = "day"), 
                           size = 1))
    sample_polys@data <- as.data.frame(sample_polys@data)
    
    ### quantify disturbance
    sample_polys@data$mort_0 <- 0.0 # no disturbance
    sample_polys@data$mort_1 <- 0.0 # harvest, thinning
    sample_polys@data$mort_2 <- 0.0 # biotic, e.g. bark beetle
    sample_polys@data$mort_3 <- 0.0 # abiotic disturbance: drought, lightning
    sample_polys@data$mort_4 <- 0.0 # gravitational event, uprooting, storm, windthrow
    sample_polys@data$mort_6 <- 0.0 # fire
    sample_polys@data$mort_7 <- 0.0
    sample_polys@data$mort_8 <- 0.0
    sample_polys@data$mort_9 <- 0.0
    
    ### add plotID
    sample_polys@data$plotID <- paste0(
      substr(sapply(strsplit(list_tiles[par_iterator], "/"), tail, 1), 
             1, nchar(sapply(strsplit(list_tiles[par_iterator], "/"), tail, 1))-4), 
      '_', 1:length(sample_polys))
    
    ### add country information
    sample_polys@data$cntry <- cntry_string
    
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
    writeOGR(obj=sample_polys, dsn=paste0("1_raw_datasets/senf/shapes/senf_", 
                                          substr(sapply(strsplit(list_tiles[par_iterator], "/"), tail, 1), 
                                                 1, nchar(sapply(strsplit(list_tiles[par_iterator], "/"), tail, 1))-4), 
                                          '_', par_iterator), 
             layer=paste0(
               "1_raw_datasets/senf/shapes/senf_", 
               substr(sapply(strsplit(list_tiles[par_iterator], "/"), tail, 1), 
                      1, nchar(sapply(strsplit(list_tiles[par_iterator], "/"), tail, 1))-4), 
               '_', par_iterator, '.gpkg'), 
             driver="GPKG")
  }
  
  ### end of large foreach loop
}

parallel::stopCluster(cl = my.cluster)

### put together the data
### merge/bind all polygons (created as sampling areas from the different tiles)
lst <- list.files("path/to/senf/shapes", full.names = TRUE, pattern = 'senf')

### bind together
bound <- do.call(rbind, lapply(lst, vect))
bound

### assign unique plotID per sample
bound$plotID <- paste0('senf_', bound$cntry, '_', 1:nrow(bound), '_', gsub('-', '_', bound$date))

### improve destination folder and write to disk
writeVector(bound, "2_shapes/senf_data_samples.gpkg")

ger <- bound[bound$cntry == "germany"]
writeVector(ger, "2_shapes/senf_data_samples_germany.gpkg")

### remove temp folders
unlink("1_raw_datasets/senf/shapes", force = TRUE, recursive = TRUE)