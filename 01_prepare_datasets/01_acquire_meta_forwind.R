### FORWIND wind damage database

### this script gets as input the shape file mentioned above
### and outputs polygons including disturbance/date/etc information
### for a set of countries in Central Europe

# Specify your packages
my_packages <- c("terra", "raster", "sf", "tidyverse", "rgdal", "rgeos", 
                 "lubridate", "tidyr")
# Extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , "Package"])]
# Install not installed packages
if(length(not_installed)) install.packages(not_installed)

### random seeds
set.seed(123)

### imports
library(tidyverse)
require(raster)
require(rgdal)
library(lubridate)
library(tidyr)
require(rgeos)
library(sf)
require(terra)


###############################################
############### data preparation ##############
###############################################

### set working directory
setwd('working_directory')

### read data
dat <- readOGR(dsn = "1_raw_datasets/forwind_database", layer = "FORWIND_v2")

### exclude countries
dat <- dat[dat$Country %in% c('DE'), ]

### exclude irrelevant columns
dat <- dat[, !(colnames(dat@data) %in% c("Dataprovid", "Source"))]

### set date column: randomly pick a date between event date and 0.5 years after the actual event
dat@data$EventDate <- gsub('/', '-', as.character(dat@data$EventDate))
dat@data$date <-as.Date(dat@data$EventDate)

## randomly assign dates as explained above
dat@data <- dat@data %>% rowwise() %>%  mutate(date = sample(x = seq(from = as.Date(date),
                                                                     to = as.Date(date) %m+% months(6),
                                                                     by = "day"),
                                                             size = 1))
dat@data <- as.data.frame(dat@data)

# about half of the dates are now after "today" -> change to "today" (2022-10-12)
sum(dat@data$date > as.Date("2022-10-12"))

### assign plotID
colnames(dat@data)[1] <- 'plotID'
dat@data$plotID <- paste0('forwind_', dat@data$plotID, '_', substr(as.character(dat@data$date), 1, 4))

### remove unnecessary columns
dat@data <- dat@data[, !(colnames(dat@data) %in% c('size'))]

### assign disturbance agent and severity
dat@data$mort_0 <- 0.0 # no disturbance
dat@data$mort_1 <- 0.0 # harvest, thinning
dat@data$mort_2 <- 0.0 # biotic, e.g. bark beetle
dat@data$mort_3 <- 0.0 # abiotic disturbance: drought, lightning
dat@data$mort_4 <- dat@data$Damage_deg # gravitational event, uprooting, storm, windthrow
dat@data$mort_5 <- 0.0 # unknown
dat@data$mort_6 <- 0.0 # fire
dat@data$mort_7 <- 0.0
dat@data$mort_8 <- 0.0
dat@data$mort_9 <- 0.0

### assume that damage degree = -999 = 1
dat@data$mort_4[dat@data$mort_4 == -999] <- 1.0

### sort columns
dat@data <- dat@data %>% dplyr::select(plotID, date, mort_0, mort_1, mort_2, mort_3, mort_4,
                                                     mort_5, mort_6, mort_7, mort_8, mort_9, everything())

### reproject to EPSG:3035 for further analyses
dat <- st_as_sf(dat)
dat <- st_transform(dat, "EPSG:3035")
dat <- as(dat, 'Spatial')

### rename columns
colnames(dat@data)[which(colnames(dat@data) == 'Area')] <- 'area'
colnames(dat@data)[which(colnames(dat@data) == 'Country')] <- 'cntry'

### get country and region information
countries <- vect("/home/cangaroo/christopher/future_forest/forest_decline/data/2_shapes/forest_mask_force/NUTS_RG_01M_2021_3035.shp")
countries <- countries[countries$LEVL_CODE == 1, ]
countries <- st_as_sf(countries)
countries <- countries[, c(4)]

### add country and region information to polygons
dat <- st_as_sf(dat)
dat <- st_join(st_make_valid(dat), st_make_valid(countries), largest = TRUE)

# convert dates to character
dat$date <- as.character(dat$date)

### split by country packages and save to disk
ger <- dat[dat$cntry == 'DE', ]
ger$cntry <- 'germany'
ger <- vect(ger)
writeVector(ger, "2_shapes/forwind_data_samples_germany.gpkg", overwrite = TRUE)

### write complete file to disk
writeOGR(obj=dat, dsn="2_shapes/forwind_plots_epsg3035", 
         layer="forwind_plots_epsg3035.gpkg", driver="GPKG")