##### this code prepares the labels of the FNEWS dataset
##### output is the meta file, not the time series

### imports
require(terra)
require(sf)
require(exactextractr)
require(dplyr)
require(lubridate)
library(future)
library(future.apply)
library(furrr)

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

### get coverage fraction and polygonize the result for tiles/subsets
cover_frac_and_polygonize <- function(main_iter, fls, shape, path){
  
  rst <- rast(fls[main_iter])$BLUE
  values(rst) <- NA
  
  ### crop raster to minimize computational demand
  rst <- terra::crop(rst, ext(shape))
  
  ### intersect with tile extent to reduce computational load
  shape <- shape[ext(rst)]
  
  ### get coverage fraction of damage polys
  dmg1 <- aggregate(shape[shape$Schadtyp == 1])
  dmg2 <- aggregate(shape[(shape$Schadtyp == 2) | (shape$Schadtyp == 22) | (shape$Schadtyp == 23)])
  dmg6 <- aggregate(shape[shape$Schadtyp == 6])
  dmg10 <- aggregate(shape[shape$Schadtyp == 10])
  
  ls <- list()
  
  if (nrow(dmg1) > 0){
    rst1 <- coverage_fraction(rst, st_as_sf(dmg1))
    rst1 <- rast(rst1)
    names(rst1) <- "mort_4" # wind 
    ls <- c(ls, rst1)
  }
  if (nrow(dmg2) > 0){
    rst2 <- coverage_fraction(rst, st_as_sf(dmg2))
    rst2 <- rast(rst2)
    names(rst2) <- "mort_2" # bark beetle
    ls <- c(ls, rst2)
  }
  if (nrow(dmg6) > 0){
    rst6 <- coverage_fraction(rst, st_as_sf(dmg6))
    rst6 <- rast(rst6)
    names(rst6) <- "mort_5" # unknown
    ls <- c(ls, rst6)
  }
  if (nrow(dmg10) > 0){
    rst10 <- coverage_fraction(rst, st_as_sf(dmg10))
    rst10 <- rast(rst10)
    names(rst10) <- "healthy" # healthy
    ls <- c(ls, rst10)
  }
  rst <- rast(ls)
  
  
  ### remove pixels too close to forest edge by negative buffer
  ### from CLMS Forest Cover (as in the other scripts)
  
  ### determine if a raster cell is a forest pixel (incl buffer) or not
  ### mask the raster cells which are not forest
  forest <- rast(paste0(path, '/1_raw_datasets/copernicus_forest_layers/germany/DATA/germany.vrt'))
  forest <- crop(forest, ext(rst))
  forest[forest == 0, ] <- NA
  forest[forest == 1, ] <- 0
  forest[forest == 2, ] <- 0
  forest[forest > 2, ] <- NA
  
  forest <- as.polygons(forest, dissolve = TRUE, na.rm = TRUE)
  forest <- aggregate(forest)
  forest <- vect(st_buffer(st_as_sf(forest), -15))
  forest <- project(forest, crs(rst))
  rst <- terra::mask(rst, forest)
  
  ### also mask by damage polygons (we need only damage pixels, because we only have presence data)
  rst <- terra::mask(rst, aggregate(shape))
    
  ### polygonize
  r <- rast(ncols = 2, nrows = 2)
  values(r) <- 1:ncell(r)
  crs(r) <- crs(rst)
  ext(r) <- ext(rst)
  
  makeTiles(rst, r, filename=paste0(path, "/1_raw_datasets/fnews/tmp/", shape$prefix[1], "_", main_iter, "_tile_.tif"), extend=FALSE, na.rm=FALSE, overwrite=TRUE)
  rst1 <- rast(paste0(path, "/1_raw_datasets/fnews/tmp/", shape$prefix[1], "_", main_iter, "_tile_1.tif"))
  rst2 <- rast(paste0(path, "/1_raw_datasets/fnews/tmp/", shape$prefix[1], "_", main_iter, "_tile_2.tif"))
  rst3 <- rast(paste0(path, "/1_raw_datasets/fnews/tmp/", shape$prefix[1], "_", main_iter, "_tile_3.tif"))
  rst4 <- rast(paste0(path, "/1_raw_datasets/fnews/tmp/", shape$prefix[1], "_", main_iter, "_tile_4.tif"))
  pols1 <- as.polygons(rst1, dissolve = FALSE, values = TRUE, 
                       na.rm = TRUE, trunc = FALSE)
  pols2 <- as.polygons(rst2, dissolve = FALSE, values = TRUE, 
                       na.rm = TRUE, trunc = FALSE)
  pols3 <- as.polygons(rst3, dissolve = FALSE, values = TRUE, 
                       na.rm = TRUE, trunc = FALSE)
  pols4 <- as.polygons(rst4, dissolve = FALSE, values = TRUE, 
                       na.rm = TRUE, trunc = FALSE)
  pols <- rbind(pols1, pols2, pols3, pols4)
  
  ### initialize columns if they do not exist yet
  if (!("healthy" %in% names(pols))){
    pols$healthy <- 0.0
  }
  if (!("mort_2" %in% names(pols))){
    pols$mort_2 <- 0.0
  }
  if (!("mort_4" %in% names(pols))){
    pols$mort_4 <- 0.0
  }
  if (!("mort_5" %in% names(pols))){
    pols$mort_5 <- 0.0
  }
  
  ### drop those pixels that are not clearly defined as healthy (polys$healthy != 0 and != 1)
  ### or not completely covered by reference polygons
  pols <- pols[(pols$healthy == 0) | (pols$healthy == 1) | ((pols$healthy > 0) & (round(pols$mort_2 + pols$mort_4 + pols$mort_5 + pols$healthy, 0) == 1))]
  
  ### get metadata from original shapefile
  pols <- pols %>% st_as_sf() %>% st_join(st_as_sf(shape), largest = TRUE) %>% vect()
  
  forest <- rast('1_raw_datasets/copernicus_forest_layers/germany/DATA/germany.vrt')
  pols <- project(pols, crs(forest))
  forest <- crop(forest, ext(pols))
  
  forest[forest == 0, ] <- NA
  forest[forest == 1, ] <- 0
  forest[forest == 2, ] <- 1
  forest[forest > 2, ] <- NA
  
  ### note that 1 = coniferous, 0 = broadleaved
  df <- exact_extract(forest, st_as_sf(pols), progress = FALSE, fun = 'mean') # , append_cols = c('plotID')
  pols$frac_coniferous <- df
  
  ### reproject to EPSG:3035
  pols <- project(pols, crs(shape))
  
  ### add country information
  pols$cntry <- "germany"
  pols$dataset <- paste0("fnews_", shape$prefix[1])
  
  ### mortality classes
  pols$healthy <- round(pols$healthy, 2)
  pols$mort_2 <- round(pols$mort_2, 2)
  pols$mort_4 <- round(pols$mort_4, 2)
  pols$mort_5 <- round(pols$mort_5, 2)
  
  ### clip rounding errors
  pols$mort_2 <- ifelse(pols$mort_2 > 1, 1, ifelse(pols$mort_2 < 0, 0, pols$mort_2))
  pols$mort_4 <- ifelse(pols$mort_4 > 1, 1, ifelse(pols$mort_4 < 0, 0, pols$mort_4))
  pols$mort_5 <- ifelse(pols$mort_5 > 1, 1, ifelse(pols$mort_5 < 0, 0, pols$mort_5))
  
  ### sometimes, reference polygons overlap so that mortality is much higher than 100%
  ### in this case we need to prioritize a damage class
  ### initialize mort_0 as 1 (= 0% damage)
  pols$mort_0 <- pols$healthy
  ### use damage information to add damage to mort_0
  pols$mort_0 <- 1.0 - pols$mort_2 - pols$mort_4 - pols$mort_5
  pols$mort_0[pols$healthy == 1] <- 1.0
  pols$mort_0 <- round(pols$mort_0, 2)
  pols$mort_0 <- ifelse(pols$mort_0 > 1, 1, ifelse(pols$mort_0 < 0, 0, pols$mort_0))
  pols$mort_1 <- 0.0
  pols$mort_3 <- 0.0
  pols$mort_6 <- 0.0
  pols$mort_7 <- 0.0
  pols$mort_8 <- 0.0
  pols$mort_9 <- 0.0
  
  ### somtimes, the polygons do not contain a date -> remove them
  pols <- pols[(!(is.na(pols$Schadensdatum) & is.na(pols$Referenzdatum)))]
  
  ### removing the pixels without date information might leave an empty SpatVector -> stop iteration
  if (nrow(pols) > 0){
    pols$earliest <- ifelse(is.na(pols$Schadensdatum), pols$Referenzdatum, pols$Schadensdatum)
    
    ### assign random date up to six months after earliest date
    pols_df <- as.data.frame(pols)
    pols_df <- pols_df %>%
      rowwise() %>%  mutate(date = sample(x = seq(from = as.Date(earliest, format = "%Y-%m-%d"),
                                                  to = as.Date(earliest, format = "%Y-%m-%d") %m+% months(6),
                                                  by = "day"),
                                          size = 1))
    pols$date <- as.character(pols_df$date)
    
    pols$date <- as.character(pols$date)
    writeVector(pols, paste0("1_raw_datasets/fnews/tmp/", shape$prefix[1], "_", main_iter, ".gpkg"), overwrite = TRUE)
    ### end of if nrow(pols) > 0
  }
  
  # ### clean up
  # file.remove(paste0(path, "/1_raw_datasets/fnews/tmp/", shape$prefix[1], "_", main_iter, "_tile_*.tif"))
  
  ### end of function cover_frac_and_polygonize
}

### this function finishes the dataset preparation:
### each prefix (nisa, bw1, bw2, sax) will get one metafile
finish_dataset_preparation <- function(main_iter, prefixes){
  
  prefix <- prefixes[main_iter]
  
  ### read and merge the polys
  fls <- list.files('path/to/1_raw_datasets/fnews/tmp', 
                    pattern = paste0(prefix, ".*gpkg$"), full.names = TRUE)
  polys <- do.call(rbind, lapply(fls, vect))
  
  ### sanity check
  sum(round(polys$mort_0 + polys$mort_2 + polys$mort_4 + polys$mort_5, 1) == 1)  == nrow(polys)
  
  ### assign plotID
  ### use naming convention as in the other datasets
  polys$plotID <- paste0(polys$dataset, "_", 1:nrow(polys), '_', gsub('-', '_', as.character(polys$date)))
  
  ### reorder dataframe columns
  polys_df <- as.data.frame(polys)
  polys_df <- polys_df %>% dplyr::select(plotID, date, mort_0, mort_1, mort_2, mort_3, mort_4,
                                         mort_5, mort_6, mort_7, mort_8, mort_9, everything())
  values(polys) <- polys_df
  
  polys$date <- as.character(polys$date)
  writeVector(polys, paste0('2_shapes/', polys$dataset[1], '.gpkg'), overwrite = TRUE)
  
  ### end of function finish_dataset_preparation
}

### set working directory
path <- ""
setwd(path)

### load data
shape <- vect("1_raw_datasets/fnews/referenzdaten_fnews_3_0_one_poly_erased_merged.gpkg")


### filter for minimum of 10m height or unknown
shape <- shape[(shape$Bestandeshoehe > 10) | is.na(shape$Bestandeshoehe)]

table(shape$Schadtyp)
# mostly bark beetle (no green attacks, but red and grey attacks),
# windthrow, or unknown

table(shape$Aufarbeitungszustand)
# primarily "uncleared", second-most abundant is "cleared"

table(shape$Schadensausmass)
# in 2/3 of cases >=91% damage area
### filter those that are either 0 or at least 91%
shape <- shape[(shape$Schadensausmass == 0) | (shape$Schadensausmass == 10) | ((shape$Schadtyp == 10) & is.na(shape$Schadensausmass))]

### define minimum extent
shape$area <- expanse(shape)
shape <- shape[shape$area > 100]

table(shape$Referenz)
# almost always based on aerial imagery

### reproject to EPSG:3035
shape <- project(shape, "EPSG:3035")

### check which FORCE datacube tiles are relevant
cube <- vect('1_raw_datasets/copernicus_forest_layers/germany/datacube-grid_DEU.gpkg')
cube <- cube[, 1]
cube <- project(cube, "EPSG:3035")
shape <- shape %>% st_as_sf() %>% st_join(st_as_sf(cube), largest = TRUE) %>% vect()

### there are four main areas: turn them into four aoi's
### we can intersect them with the "countries" shape to make them easily separable
### get country and region information
countries <- vect("/home/cangaroo/christopher/future_forest/forest_decline/data/2_shapes/forest_mask_force/NUTS_RG_01M_2021_3035.shp")
countries <- countries[countries$LEVL_CODE == 1, ]
countries <- st_as_sf(countries)
countries <- countries[, c(4)]

shape <- shape %>% st_as_sf() %>% st_join(countries, largest = TRUE) %>%  vect()
names(shape)[which(names(shape) == "NAME_LATN")] <- "region"

### use our own shapefile to separate into 4 areas
datasets <- vect("1_raw_datasets/fnews/dataset_indicator.gpkg")
datasets <- project(datasets, crs(shape))
datasets <- st_as_sf(datasets)[, c(2)]
shape <- shape %>% st_as_sf() %>% st_join(datasets, largest = TRUE) %>%  vect()

nisa <- shape[shape$dataset == "nisa"]
bw1 <- shape[shape$dataset == "bw1"]
bw2 <- shape[shape$dataset == "bw2"]
sax <- shape[shape$dataset == "sax"]

### load all rsters covering the relevant area
table(nisa$Tile_ID)
### nisa: X0060_Y0045, X0061_Y0045, X0062_Y0045,
###       X0060_Y0046, X0061_Y0046, X0062_Y0046, X0063_Y0046
###       X0061_Y0047, X0062_Y0047, X0063_Y0047
table(sax$Tile_ID)
table(bw1$Tile_ID)
table(bw2$Tile_ID)

tile_ids <- c(unique(nisa$Tile_ID), unique(sax$Tile_ID), unique(bw1$Tile_ID), unique(bw2$Tile_ID))

nisa$prefix <- 'nisa'
sax$prefix <- 'sax'
bw1$prefix <- 'bw1'
bw2$prefix <- 'bw2'

shape <- rbind(nisa, sax, bw1, bw2)

### get list of files
TILE_BASE_PATH <- '/media/cangaroo/Elements1/christopher/future_forest/forest_decline/force/level2/germany/'
fls <- lapply(tile_ids, function(tile){
  files <- paste0(
    TILE_BASE_PATH, 
    tile, 
    '/', 
    list.files(paste0(TILE_BASE_PATH, tile, '/'), pattern = "SEN2")[1])
  files
})
### unlist
fls <- unlist(fls)

### create tmp folder for saving intermediate results
dir.create(paste0(getwd(), "/1_raw_datasets/fnews/tmp/"), recursive = TRUE)

plan(multicore, workers = length(fls))
future_map(1:length(fls), function(main_iter){cover_frac_and_polygonize(main_iter, fls, shape, path)})

# stop the future plan
plan('sequential')

##### finish the dataset preparation for all of the four areas
prefixes <- c("nisa", "bw1", "bw2", "sax")

plan(multicore, workers = length(prefixes)) # this seems to do something in Terminal, but not in RStudio

future_map(1:length(prefixes), function(main_iter){finish_dataset_preparation(main_iter, prefixes)})

# stop the future plan
plan('sequential')

print("Script finished")