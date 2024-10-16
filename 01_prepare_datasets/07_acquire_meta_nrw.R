# Specify your packages
my_packages <- c('terra', 'raster', 'sp', 'sf', 'lubridate', 'rgeos', 
                 'rgdal', 'tidyverse', 'exactextractr')
# Extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , 'Package'])]
# Install not installed packages
if(length(not_installed)) install.packages(not_installed)

### random seed
set.seed(123)

### imports
require(raster)
require(sf)
require(sp)
require(rgeos)
require(lubridate)
require(tidyverse)
require(terra)
require(exactextractr)
require(foreach)
require(doParallel)
require(data.table)

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


###############################################################################
######## prepare the data with the goal of retrospective analysis #############
######## + minimal extent, considering non-disturbance data as well ############
######## the resulting dataset contains ALL informatin necessary #############
###############################################################################

### set working directory
path = ''
setwd(path)

### read rasters (aerial images of the study area)
rast1 <- rast('path/to/dop10rgbi_32_406_5685_1_nw_2021.jp2')
rast2 <- rast('path/to/dop10rgbi_32_406_5686_1_nw_2021.jp2')
rast3 <- rast('path/to/dop10rgbi_32_407_5685_1_nw_2021.jp2')
rast4 <- rast('path/to/dop10rgbi_32_407_5686_1_nw_2021.jp2')
names(rast2) <- names(rast1)
names(rast3) <- names(rast1)
names(rast4) <- names(rast1)
aerial <- merge(rast1, rast2, rast3, rast4)

### read input shapefile containing the disturbance polygons
shape = vect(paste0(path, 'path/to/deadtrees_nrw.gpkg'))

### read German vrt example file
rast <- rast('path/to/X0056_Y0047/20150802_LEVEL2_SEN2A_BOA.tif')$BLUE
shape <- terra::project(shape, crs(rast))
aerial <- terra::project(aerial, crs(shape))
rast <- terra::crop(rast, ext(aerial))
values(rast) <- 1

### mask by forest mask
forest <- rast('1_raw_datasets/copernicus_forest_layers/germany/DATA/germany.vrt')
forest <- terra::project(forest, crs(rast))
forest <- crop(forest, ext(rast))

forest[forest == 0, ] <- NA
forest[forest == 1, ] <- 0
forest[forest == 2, ] <- 0
forest[forest > 2, ] <- NA

forest <- as.polygons(forest, dissolve = TRUE, na.rm = TRUE)
forest <- vect(st_buffer(st_as_sf(forest), -15))

### mask by negatively buffered forest mask
rast <- terra::mask(rast, forest)

### get disturbance categories
needle <- shape[shape$dmg_cat == 1 | shape$dmg_cat == 3]
broadleave <- shape[shape$dmg_cat == 2]
infrastructure <- shape[shape$dmg_cat == 5]
clearcut <- shape[shape$dmg_cat == 6]
regrowth <- shape[shape$dmg_cat == 7]
glade <- shape[shape$dmg_cat == 8]
soil <- shape[shape$dmg_cat == 9]

### aggregate the disturbance polygons
needle <- aggregate(needle)
broadleave <- aggregate(broadleave)
clearcut <- aggregate(clearcut)
regrowth <- aggregate(regrowth)
soil <- aggregate(soil)

### get coverage fraction for each raster cell covered by polygon
needle_rast <- coverage_fraction(rast, st_as_sf(needle))
broadleave_rast <- coverage_fraction(rast, st_as_sf(broadleave))
clearcut_rast <- coverage_fraction(rast, st_as_sf(clearcut))
regrowth_rast <- coverage_fraction(rast, st_as_sf(regrowth))
soil_rast <- coverage_fraction(rast, st_as_sf(soil))

### combine the results
rast_dat <- c(rast(needle_rast), rast(broadleave_rast), rast(clearcut_rast), rast(regrowth_rast), 
              rast(soil_rast)) # , rast(loose_rast)
names(rast_dat) <- c('mort_con', 'mort_dec', # 'mort_loose', 
                     'mort_cleared', 'mort_regrowth', 'mort_soil')

### convert to polys
polys <- as.polygons(rast_dat, dissolve = FALSE, values = TRUE, 
                     na.rm = TRUE, trunc = FALSE)

### define the disturbance categories
polys$mort_1 <- polys$mort_con + polys$mort_dec
polys$mort_5 <- polys$mort_cleared

### remove edges to get rid of edge effects
polys <- polys[forest]

### drop regrowth (we don't use it in the study and it is not clearly defined)
polys <- polys[!(polys$mort_regrowth > 0)]

### drop the edges of study area to avoid wrongly labelled pixels
rlt <- is.related(polys, vect(st_buffer(st_as_sf(vect(ext(aerial))), -15)), 'coveredby')
polys <- polys[rlt]
aoi <- vect('path/to/aoi_nrw.gpkg')
aoi <- aggregate(aoi)
aoi <- vect(st_buffer(st_as_sf(aoi), -15))
polys <- polys[aoi]
polys_ext <- aggregate(polys)
polys_ext <- vect(st_buffer(st_as_sf(polys_ext), -15))
polys <- polys[polys_ext]

### assign metadata
polys$date <- as.character('2021-06-14')
polys$plotID <- paste0('nrw_', 1:(nrow(polys)), '_', gsub('-', '_', as.character(polys$date)))

polys$dataset <- 'nrw'
polys$mort_0 <- 1.0 - polys$mort_1 - polys$mort_cleared - polys$mort_soil
polys$mort_0[polys$mort_0 < 0] <- 0
polys$mort_2 <- 0.0
polys$mort_3 <- 0.0
polys$mort_4 <- 0.0
polys$mort_6 <- 0.0
polys$mort_7 <- 0.0
polys$mort_8 <- 0.0
polys$mort_9 <- 0.0
polys$date <- as.character(polys$date)
polys$country <- 'germany'
### assign observation year
polys$year <- "2021"

### get country and region information
countries <- vect('/home/cangaroo/christopher/future_forest/forest_decline/data/2_shapes/forest_mask_force/NUTS_RG_01M_2021_3035.shp')
countries <- countries[countries$LEVL_CODE == 1, ]
countries <- st_as_sf(countries)
countries <- countries[, c(4)]
countries <- st_make_valid(countries)

### add country and region information to polygons
### in case of polygons spanning more than one region, 
### we only keep one region (that is enough for our purpose
polys <- st_as_sf(polys)
polys <- st_make_valid(polys)
polys <- st_join(polys, countries, largest = TRUE)
names(polys)[which(names(polys) == "NAME_LATN")] <- 'region'
polys <- vect(polys)

### get fraction coniferous from Copernicus layers
forest <- rast('1_raw_datasets/copernicus_forest_layers/germany/DATA/germany.vrt')
forest <- crop(forest, ext(polys))

forest[forest == 0, ] <- NA
forest[forest == 1, ] <- 0
forest[forest == 2, ] <- 1
forest[forest > 2, ] <- NA

### note that 1 = coniferous, 0 = broadleaved
df <- exact_extract(forest, st_as_sf(polys), progress = FALSE, fun = 'mean') # , append_cols = c('plotID')
polys$frac_coniferous <- df

### double-check: duplicate plotID's?
sum(duplicated(polys$plotID))

### reorder dataframe columns
polys_df <- as.data.frame(polys)
polys_df <- polys_df %>% dplyr::select(plotID, date, mort_0, mort_1, mort_2, mort_3, mort_4,
                                       mort_5, mort_6, mort_7, mort_8, mort_9, everything())
values(polys) <- polys_df

### remove infrastructure from sampling
polys <- terra::mask(polys, buffer(infrastructure, width = 0), inverse = TRUE)
polys <- terra::mask(polys, buffer(glade, width = 0), inverse = TRUE)

polys$date <- as.character(polys$date)
writeVector(polys, "2_shapes/nrw_samples.gpkg", overwrite = TRUE)