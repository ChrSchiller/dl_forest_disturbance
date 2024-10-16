#### Schiefer dataset
### the script outputs disturbance data from the Felix Schiefer dataset
### only a metafile is written to disk, S2 time series will be acquired later


# Specify your packages
my_packages <- c('terra', 'raster', 'sp', 'sf', 'lubridate', 'rgeos', 
                 'rgdal', 'tidyverse', 'exactextractr', 'tidyterra')
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

### set working directory
path = ''
setwd(path)


shapes <- vect("path/to/schiefer.gpkg")
shapes$plotID <- substr(shapes$layer, 27, 32)
shapes$year <- substr(shapes$layer, 15, 18)

### join with date from csv file
dates <- read.table("path/to/deadwood_sites_schiefer.csv", sep = ";", header = TRUE)
head(dates)
dates <- dates[, c("Plot", "acquisition_date", "year")]
colnames(dates)[c(1, 2)] <- c("plotID", "date")
dates$date <- as.Date(dates$date, format = "%d.%m.%Y")
dates$year <- as.character(dates$year)

### join date information
shapes_df <- as.data.frame(shapes)
shapes_df <- left_join(shapes_df, dates, by = c("plotID", "year"))
values(shapes) <- shapes_df

### Finland is not part of our area of interest -> drop it
shapes <- shapes[!(shapes$plotID %in% c("FIN001", "FIN002", "FIN003")), ]

### aggregate by plotID and year
dissolved <- aggregate(shapes, by = c('plotID', 'year'))

# take rast example for the time: X0052_Y0053
rast <- rast('/mnt/storage/forest_decline/force/higher_level/germany_forest_mask_full/mosaic/2015-2022_001-365_HL_TSA_SEN2L_STACK_TSS_20150819.vrt')$BLU

dissolved <- terra::project(revisit, crs(rast))

rast <- terra::crop(rast, ext(dissolved))
rast <- terra::mask(rast, dissolved)

### differentiate by year of observation
y2017 <- aggregate(dissolved[dissolved$year == '2017'])
# y2017 <- st_as_sf(y2017[, names(y2017) %in% c("plotID", "year", "date")])
y2018 <- aggregate(dissolved[dissolved$year == '2018'])
# y2018 <- st_as_sf(y2018[, names(y2018) %in% c("plotID", "year", "date")])
y2019 <- aggregate(dissolved[dissolved$year == '2019'])
# y2019 <- st_as_sf(y2019[, names(y2019) %in% c("plotID", "year", "date")])
y2020 <- aggregate(dissolved[dissolved$year == '2020'])
### get coverage fraction for each raster cell covered by polygon
year2017 <- coverage_fraction(rast, st_as_sf(y2017))
year2018 <- coverage_fraction(rast, st_as_sf(y2018))
year2019 <- coverage_fraction(rast, st_as_sf(y2019))
year2020 <- coverage_fraction(rast, st_as_sf(y2020))
### there are no revisits in 2021

### now combine everything and get total damage
dissolved <- aggregate(dissolved)
dmg_total <- coverage_fraction(rast, st_as_sf(dissolved))

### combine the results
rast_dat <- c(rast(year2017), rast(year2018), rast(year2019), rast(year2020), rast(dmg_total))
names(rast_dat) <- c('mort_year_2017', 'mort_year_2018','mort_year_2019', 'mort_year_2020', 'mort_3')


### get bbox by plotID and then use it as mask in following code
bboxes <- aggregate(revisit, by = c("plotID"))
boxes <- vect(terra::ext(bboxes[1, ]))
for (iter in 2:nrow(bboxes)){
  boxes <- rbind(boxes, vect(terra::ext(bboxes[iter, ])))
}
crs(boxes) <- crs(polys_mort_3_dist)

## polygonize and dissolve all pixels with at least 0.01 dmg_total (= mort_3)
polys_mort_3 <- as.polygons(crop(rast_dat$mort_3, boxes[1, ]), dissolve = FALSE, values = TRUE, 
                            na.rm = TRUE, trunc = FALSE)

for (iter in 2:nrow(boxes)){
  polys_mort_3_next <- as.polygons(crop(rast_dat$mort_3, boxes[iter, ]), dissolve = FALSE, values = TRUE, 
                                   na.rm = TRUE, trunc = FALSE)
  polys_mort_3 <- rbind(polys_mort_3, polys_mort_3_next)
}
polys_mort_3_dist <- polys_mort_3[values(polys_mort_3) > 0.01]


### intersect polys_mort_3_dist (total damage final samples) with the other polys
### to get all the disturbance values for each year
extracted <- terra::extract(rast(year2017), polys_mort_3_dist, fun="mean")
polys_mort_3_dist$mort_year_2017 <- extracted[, 2]
extracted <- terra::extract(rast(year2018), polys_mort_3_dist, fun="mean")
polys_mort_3_dist$mort_year_2018 <- extracted[, 2]
extracted <- terra::extract(rast(year2019), polys_mort_3_dist, fun="mean")
polys_mort_3_dist$mort_year_2019 <- extracted[, 2]
extracted <- terra::extract(rast(year2020), polys_mort_3_dist, fun="mean")
polys_mort_3_dist$mort_year_2020 <- extracted[, 2]

### finished acquisition of disturbance in Schiefer/Black Forest
### concatenate all the resulting samples
polys_all <- polys_mort_3_dist # just for convenience

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
# get plot ID using st_join
polys_all <- st_as_sf(polys_all)
# multipart to single parts with boxes variable
boxes_single <- st_cast(st_as_sf(boxes), 'POLYGON')
# st_join with largest = TRUE with shapes
boxes_single <- st_join(boxes_single, st_as_sf(revisit[, c("plotID")]), largest = TRUE)
colnames(boxes_single)[1] <- 'plot'
### for the spatial join, we need the bounding boxes of the plot areas including plot ID
polys_all <- st_join(polys_all, st_buffer(boxes_single, 50), largest = TRUE)

# get observation date using left_join and plot ID 
# make sure to get the last observation for each date (can be 2019 as well)
# join date by plotID
# get latest date by plotID
dates_last <- dates %>% 
  group_by(plotID) %>%
  slice(which.max(as.Date(date, '%Y-%m-%d')))
dates_last <- as.data.frame(dates_last)
dates_last <- dates_last[, c("plotID", "date", "year")]
colnames(dates_last)[1] <- 'plot'

dates_first <- dates %>% 
  group_by(plotID) %>%
  slice(which.min(as.Date(date, '%Y-%m-%d')))
dates_first <- as.data.frame(dates_first)
# dates_first <- dates_first[, c("plotID", "date_first_obs", "year_first_obs")]
colnames(dates_first) <- c("plot", "date_first_obs", "year_first_obs")

polys_all <- left_join(polys_all, dates_last, by = c("plot"))
polys_all <- left_join(polys_all, dates_first, by = c("plot"))

polys_all$date <- as.character(polys_all$date)

## use naming convention as in the other datasets
polys_all$plotID <- paste0('schiefer_', 1:nrow(polys_all), '_', gsub('-', '_', as.character(polys_all$date)))

### quantify disturbance
polys_all$mort_0 <- 0.0 # no disturbance
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
### in case of polygons spanning more than one region, 
### we only keep one region (that is enough for our purpose
polys_all <- st_as_sf(polys_all)
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

### check how many pixels remain when removing those that recovered between 2017 and 2020
polys_all[(polys_all$mort_year_2017 == 0) & (polys_all$year_first_obs == 2017), ]
polys_all[(polys_all$mort_year_2018 == 0) & (polys_all$year_first_obs == 2018), ]
### we have 397 samples that were totally undamaged in 2017!

nrow(polys_all[(polys_all$mort_year_2017 == 0) & (polys_all$year_first_obs == 2017) & (polys_all$mort_3 > .1), ]) + 
  nrow(polys_all[(polys_all$mort_year_2018 == 0) & (polys_all$year_first_obs == 2018) & (polys_all$mort_3 > .1), ])
### 149 out of 722 samples are completely undamaged in the first year but damaged more than 10% later on

### geopackage for validation is finished here
polys_all$date <- as.character(polys_all$date)

### write to disk
writeVector(polys_all, "2_shapes/schiefer.gpkg", overwrite = TRUE)