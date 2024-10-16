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
require(rgdal)
require(raster)
require(sf)
require(sp)
require(rgeos)
require(lubridate)
require(tidyverse)
require(terra)
require(exactextractr)

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
######## the resulting dataset contains ALL information necessary #############
###############################################################################

### set working directory
path = '/home/cangaroo/christopher/future_forest/forest_decline/data'
setwd(path)

### these are the two input shapefiles (years 2017 and 2019)
shape_2017 = vect(paste0(path, '/1_raw_datasets/selina/reference/deadtrees_2017/deadtrees_2017.shp'))
shape_2018 = vect(paste0(path, '/1_raw_datasets/selina/reference/deadtrees_2018/deadtrees_2018.shp'))
shape_2019 = vect(paste0(path, '/1_raw_datasets/selina/reference/deadtrees_2019/deadtrees_2019.shp'))
shape_2020 = vect(paste0(path, '/1_raw_datasets/selina/reference/deadtrees_2020/deadtrees_2020.shp'))

### assign observation year
shape_2017$year <- "2017"
shape_2018$year <- "2018"
shape_2019$year <- "2019"
shape_2020$year <- "2020"
shape_2019$type <- as.integer(shape_2019$type)

## mask all dieback shapes with all aoi shapes (for each year)
## so that resulting dieback areas have been scanned each year
mask_1719 <- vect(paste0(path, '/1_raw_datasets/selina/reference/deadtrees_area_2017_2019.shp'))
mask_1820 <- vect(paste0(path, '/1_raw_datasets/selina/reference/areas_18_20.shp'))
shape_2017 <- terra::mask(shape_2017, mask_1820, inverse = FALSE)
shape_2019 <- terra::mask(shape_2019, mask_1820, inverse = FALSE)
shape_2018 <- terra::mask(shape_2018, mask_1719, inverse = FALSE)
shape_2020 <- terra::mask(shape_2020, mask_1719, inverse = FALSE)

### get the overlapping area of the two observation areas: 17/19 and 18/20
mask17_20 <- crop(mask_1719, mask_1820)
mask17_20 <- crop(mask17_20, mask_1719)

## union of those files
names(shape_2018)[1] <- "type"
names(shape_2020)[1] <- "type"
merged <- rbind(shape_2017, shape_2018, shape_2019, shape_2020)

# dissolve, but keep the two forest types
dissolved <- aggregate(merged, by = c('type', 'year'))

# take rast example for the time: X0052_Y0053
rast <- rast('/mnt/storage2/forest_decline/force/higher_level/benelux_forest_mask_full/mosaic/2015-2022_001-365_HL_TSA_SEN2L_STACK_TSS_20150706.vrt')$BLU

dissolved <- terra::project(dissolved, crs(rast))

rast <- terra::crop(rast, ext(dissolved))
rast <- terra::mask(rast, dissolved)

### type = 1 are dead CONIFEROUS trees, type = 2 are dead DECIDUOUS trees
y2017 <- dissolved[dissolved$year == '2017']
y2017 <- st_as_sf(y2017[, -c(3, 4, 5)])
y2018 <- dissolved[dissolved$year == '2018']
y2018 <- st_as_sf(y2018[, -c(3, 4, 5)])
y2019 <- dissolved[dissolved$year == '2019']
y2019 <- st_as_sf(y2019[, -c(3, 4, 5)])
y2020 <- dissolved[dissolved$year == '2020']
y2020 <- st_as_sf(y2020[, -c(3, 4, 5)])
### get coverage fraction for each raster cell covered by polygon
year2017 <- coverage_fraction(rast, y2017)
year2018 <- coverage_fraction(rast, y2018)
year2019 <- coverage_fraction(rast, y2019)
year2020 <- coverage_fraction(rast, y2020)

### now combine coniferous (type == 1) damage + deciduous damage (type == 2) in 2020 to get total damage
con_all <- aggregate(dissolved[dissolved$type == 1], by = "type")
dec_2020 <- dissolved[dissolved$type == 2]
dec_2020 <- aggregate(dec_2020[dec_2020$year == 2020], by = "type")
dissolved <- union(con_all, dec_2020)
dissolved <- aggregate(dissolved)
dissolved <- st_as_sf(dissolved[, -c(2, 3, 4, 5, 6, 7)])
dmg_total <- coverage_fraction(rast, dissolved)
### now we got the correct values each year + total damage in 2020

### combine the results
rast_dat <- c(rast(year2017), rast(year2018), rast(year2019), rast(year2020), rast(dmg_total))
names(rast_dat) <- c('mort_year_2017_con', 'mort_year_2017_dec', 'mort_year_2018_con', 
                     'mort_year_2018_dec', 'mort_year_2019_con', 'mort_year_2019_dec', 'mort_year_2020_con', 'mort_year_2020_dec', 'mort_3')

### add sampling here
## polygonize and dissolve all pixels with at least 0.01 dmg_total (=mort_3)
mask17_20 <- terra::project(mask17_20, crs(rast_dat))
polys_mort_3_item1 <- as.polygons(crop(rast_dat$mort_3, mask17_20[1]), dissolve = FALSE, values = TRUE, 
                                  na.rm = TRUE, trunc = FALSE)
polys_mort_3_item2 <- as.polygons(crop(rast_dat$mort_3, mask17_20[2]), dissolve = FALSE, values = TRUE, 
                                  na.rm = TRUE, trunc = FALSE)
polys_mort_3_item3 <- as.polygons(crop(rast_dat$mort_3, mask17_20[3]), dissolve = FALSE, values = TRUE, 
                                  na.rm = TRUE, trunc = FALSE)
polys_mort_3_item4 <- as.polygons(crop(rast_dat$mort_3, mask17_20[4]), dissolve = FALSE, values = TRUE, 
                                  na.rm = TRUE, trunc = FALSE)
polys_mort_3 <- rbind(polys_mort_3_item1, polys_mort_3_item2, polys_mort_3_item3, polys_mort_3_item4)

polys_mort_3_dist <- polys_mort_3[values(polys_mort_3) > 0.01]

### intersect polys_mort_3_dist (total damage final samples) with the other polys
### to get all the disturbance values for each year
extracted <- terra::extract(rast(year2017), polys_mort_3_dist, fun="mean")
polys_mort_3_dist$mort_year_2017_con <- extracted[, 2]
polys_mort_3_dist$mort_year_2017_dec <- extracted[, 3]
extracted <- terra::extract(rast(year2018), polys_mort_3_dist, fun="mean")
polys_mort_3_dist$mort_year_2018_con <- extracted[, 2]
polys_mort_3_dist$mort_year_2018_dec <- extracted[, 3]
extracted <- terra::extract(rast(year2019), polys_mort_3_dist, fun="mean")
polys_mort_3_dist$mort_year_2019_con <- extracted[, 2]
polys_mort_3_dist$mort_year_2019_dec <- extracted[, 3]
extracted <- terra::extract(rast(year2020), polys_mort_3_dist, fun="mean")
polys_mort_3_dist$mort_year_2020_con <- extracted[, 2]
polys_mort_3_dist$mort_year_2020_dec <- extracted[, 3]

polys_mort_3_dist$mort_year_2017 <- polys_mort_3_dist$mort_year_2017_con + polys_mort_3_dist$mort_year_2017_dec
polys_mort_3_dist$mort_year_2018 <- polys_mort_3_dist$mort_year_2018_con + polys_mort_3_dist$mort_year_2018_dec
polys_mort_3_dist$mort_year_2019 <- polys_mort_3_dist$mort_year_2019_con + polys_mort_3_dist$mort_year_2019_dec
polys_mort_3_dist$mort_year_2020 <- polys_mort_3_dist$mort_year_2020_con + polys_mort_3_dist$mort_year_2020_dec
### finished acquisition of disturbance in LUX 

### now get undisturbed samples in LUX
### load forest shapefile
landuse_frst <- vect(paste0(path, '/1_raw_datasets/selina/reference/Landuse2015_3044_fixed/Landuse2015_3044_fixed_dissolved.shp'))

### intersect forest shapefile with overlap of observation areas
landuse_frst <- terra::project(landuse_frst, crs(mask17_20))
landuse_frst <- crop(landuse_frst, mask17_20)

### remove disturbed areas including buffer
landuse_frst <- terra::project(landuse_frst, crs(shape_2017))
landuse_frst <- erase(landuse_frst, buffer(shape_2017, width = 50))
landuse_frst <- erase(landuse_frst, buffer(shape_2018, width = 50))
landuse_frst <- erase(landuse_frst, buffer(shape_2019, width = 50))
landuse_frst <- erase(landuse_frst, buffer(shape_2020, width = 50))

### sample undisturbed areas as described elsewhere
rast <- rast('/mnt/storage2/forest_decline/force/higher_level/benelux_forest_mask_full/mosaic/2015-2022_001-365_HL_TSA_SEN2L_STACK_TSS_20150706.vrt')$BLU

landuse_frst <- terra::project(landuse_frst, crs(rast))
rast <- terra::crop(rast, ext(mask17_20))
rast <- terra::mask(rast, landuse_frst)

landuse_frst <- st_as_sf(landuse_frst)
landuse_frst <- shrinkIfPossible(landuse_frst, 10)
landuse_frst <- vect(landuse_frst)
landuse_frst <- aggregate(landuse_frst)

## sample points
no_pts <- nrow(polys_mort_3_dist) * 1.5 # number similar to number of disturbed polygons
landuse_frst <- st_as_sf(landuse_frst)

sample_pts <- st_sample(landuse_frst, no_pts)
points_matrix <- gWithinDistance(as(sample_pts, 'Spatial'), dist = 30, byid = TRUE)
points_matrix[lower.tri(points_matrix, diag=TRUE)] <- NA
v <- colSums(points_matrix, na.rm=TRUE) == 0
sample_pts <- sample_pts[v]
sample_pts <- vect(sample_pts)

### get those pixels which intersect with sample_points
sample_pts <- terra::project(sample_pts, crs(landuse_frst))
rast <- terra::project(rast, crs(landuse_frst))
values(rast) <- NA
sample_pts <- terra::project(sample_pts, crs(rast))

rast_pts <- terra::rasterize(sample_pts, rast)

## polygonize
## polygonize and dissolve all pixels with at least 0.01 dmg_total/mort_3
mask17_20 <- terra::project(mask17_20, crs(rast_pts))
polys_undist_item1 <- as.polygons(crop(rast_pts$last, mask17_20[1]), dissolve = FALSE, values = TRUE, 
                                  na.rm = TRUE, trunc = FALSE)
polys_undist_item2 <- as.polygons(crop(rast_pts$last, mask17_20[2]), dissolve = FALSE, values = TRUE, 
                                  na.rm = TRUE, trunc = FALSE)
polys_undist_item3 <- as.polygons(crop(rast_pts$last, mask17_20[3]), dissolve = FALSE, values = TRUE, 
                                  na.rm = TRUE, trunc = FALSE)
polys_undist_item4 <- as.polygons(crop(rast_pts$last, mask17_20[4]), dissolve = FALSE, values = TRUE, 
                                  na.rm = TRUE, trunc = FALSE)
polys_undist <- rbind(polys_undist_item1, polys_undist_item2, polys_undist_item3, polys_undist_item4)

### add metadata, e.g. mort, date...
polys_mort_3_dist
names(polys_undist) <- "mort_3"
polys_undist$mort_3 <- 0
polys_undist$mort_year_2017_dec <- 0
polys_undist$mort_year_2017_con <- 0
polys_undist$mort_year_2017 <- 0
polys_undist$mort_year_2018_dec <- 0
polys_undist$mort_year_2018_con <- 0
polys_undist$mort_year_2018 <- 0
polys_undist$mort_year_2019_dec <- 0
polys_undist$mort_year_2019_con <- 0
polys_undist$mort_year_2019 <- 0
polys_undist$mort_year_2020_dec <- 0
polys_undist$mort_year_2020_con <- 0
polys_undist$mort_year_2020 <- 0

### concatenate all the resulting samples
polys_all <- rbind(polys_undist, polys_mort_3_dist)

### get fraction coniferous from Copernicus layers
forest <- rast('1_raw_datasets/copernicus_forest_layers/germany/DATA/germany.vrt')
forest <- crop(forest, ext(polys_all))

forest[forest == 0, ] <- NA
forest[forest == 1, ] <- 0
forest[forest == 2, ] <- 1
forest[forest > 2, ] <- NA

### note that 1 = coniferous, 0 = broadleaved
df <- exact_extract(forest, st_as_sf(polys_all), progress = FALSE, fun = 'mean') # , append_cols = c('plotID')
polys_all$frac_coniferous <- df

### add plotID, add date, sort columns
polys_all$date <- as.Date('2020-09-19', format='%Y-%m-%d')
polys_all$date <- as.character(polys_all$date)

## use naming convention as in the other datasets
polys_all$plotID <- paste0('schwarz_', 1:nrow(polys_all), '_', gsub('-', '_', as.character(polys_all$date)))

### quantify disturbance
polys_all$mort_1 <- 0.0 # harvest, thinning
polys_all$mort_2 <- 0.0 # biotic, e.g. bark beetle
polys_all$mort_4 <- 0.0 # gravitational event, uprooting, storm, windthrow
polys_all$mort_5 <- 0.0 # unknown
polys_all$mort_6 <- 0.0 # fire
polys_all$mort_7 <- 0.0
polys_all$mort_8 <- 0.0
polys_all$mort_9 <- 0.0

### get country and region information
countries <- vect('/home/cangaroo/christopher/future_forest/forest_decline/data/2_shapes/forest_mask_force/NUTS_RG_01M_2021_3035.shp')
countries <- countries[countries$LEVL_CODE == 1, ]
countries <- st_as_sf(countries)
countries <- countries[, c(4)]
countries <- st_make_valid(countries)

### add country and region information to polygons
polys_all <- st_as_sf(polys_all)
polys_all <- st_make_valid(polys_all)
polys_all <- st_join(polys_all, countries, largest = TRUE)

names(polys_all)[which(names(polys_all) == "NAME_LATN")] <- 'region'
polys_all <- vect(polys_all)

### some more data preparation for future use
polys_all$mort_0 <- 1.0 - polys_all$mort_3

### reorder dataframe columns
polys_df <- as.data.frame(polys_all)
polys_df <- polys_df %>% dplyr::select(plotID, date, mort_0, mort_1, mort_2, mort_3, mort_4,
                                       mort_5, mort_6, mort_7, mort_8, mort_9, everything())
values(polys_all) <- polys_df

polys_all$year = '2020'
polys_all$cntry = "luxembourg"

### shapefile with damage polygons and mort_3 == 100% is finished here
polys_all$date <- as.character(polys_all$date)

### save to disk
writeVector(polys_all, '2_shapes/schwarz.gpkg', overwrite = TRUE)