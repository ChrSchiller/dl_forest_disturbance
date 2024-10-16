### this script takes as input a meta file with the disturbance pixels (as points)
### and outputs the metafile including one csv file per sample 
### containing the time series of the 10 bands of the Sentinel-2 data


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
require(lubridate)
require(tidyverse)
require(terra)
require(exactextractr)
require(doParallel)
require(data.table)
library(future)
library(future.apply)
library(furrr)


### function for parallel processing
get_ts_for_tile <- function(tile_iter, tiles, zones, output_path, enddate) {

    ### get number of Sentinel-2 scenes inside chip
    fls <- list.files(tiles[tile_iter], pattern = ".SEN2(A|B)_BOA.tif$", full.names = TRUE)

    ### subset by start and end date
    years <- lapply(fls, substr, nchar(fls)-28, nchar(fls)-21)
    # convert years to numeric
    years <- as.numeric(years)
    # filter filenames based on the condition
    fls <- subset(fls, years <= enddate)
    ### remove possible NA values in list
    fls <- fls[!is.na(fls)]

    ### if more than 0 do (continue only if there are files)
    if (length(fls) > 0){

      ### load first raster in files
      stck <- rast(fls[1])

      ### subset zones by extent of this rast
      zones_cropped <- terra::crop(zones, ext(stck), ext = TRUE)

      if (nrow(zones_cropped) > 1){

        ### divide stacking into 5 chunks
        chunks <- floor(seq.int(1, length(fls), length.out = 5))

        for (chunk_iter in 1:(length(chunks)-1)){
          print(paste0("chunk_iter = ", chunk_iter))

          ### initialize empty list for stacking
          stck <- list()

          ### subset the sentinel-2 scenes list
          fls_subset <- fls[chunks[chunk_iter]:chunks[chunk_iter+1]]

          ### loop over scenes/timesteps:
          for (iter in 1:length(fls_subset)){
            ### load the raster
            rast <- rast(fls_subset[iter])

            ### rename bands by timestamp and band name
            names(rast) <- bands
            ### get date
            date <- substr(fls_subset[iter], nchar(fls_subset[iter])-28, nchar(fls_subset[iter])-21)
            ### add date to rast names in correct order
            names(rast) <- paste0(names(rast), "_", date)

            ### stack rasters
            stck[[iter]] <- rast

            ### end of for loop stacking the rasters as list
          }

          ### cast rasters
          stck <- rast(stck)

          ### add plotID
          ### the buffer is absolutely necessary because otherwise the plotID's are not retained
          ### they would be turned into integers
          zones_rast <- terra::rasterize(buffer(zones_cropped, width = 1), stck[[1]], field = "plotID")
          stck <- c(zones_rast, stck)

          ### convert polys to points 
          ### (this is possible only if pixels are identical with polygons)
          ### as is the case with the finetuning data
          ### approach is very similar to the one for pre-training data
          points <- zones_cropped # only for convenience

          ### extract values from raster stack
          extracted <- terra::extract(stck, points, ID = FALSE)

          ### save to disk
          fwrite(extracted, paste0(
            output_path, "tmp/extracted", chunk_iter, "_",
            substr(tiles[tile_iter], nchar(tiles[tile_iter])-10, nchar(tiles[tile_iter])), ".txt")
          )

          ### end of for (chunk_iter in 1:(length(chunks)-1)){
        }

        ### load all data
        dat1 <- fread(paste0(
          output_path, "tmp/extracted1", "_",
          substr(tiles[tile_iter], nchar(tiles[tile_iter])-10, nchar(tiles[tile_iter])), ".txt")
        )
        dat2 <- fread(paste0(
          output_path, "tmp/extracted2", "_",
          substr(tiles[tile_iter], nchar(tiles[tile_iter])-10, nchar(tiles[tile_iter])), ".txt")
        )
        dat3 <- fread(paste0(
          output_path, "tmp/extracted3", "_",
          substr(tiles[tile_iter], nchar(tiles[tile_iter])-10, nchar(tiles[tile_iter])), ".txt")
        )
        dat4 <- fread(paste0(
          output_path, "tmp/extracted4", "_",
          substr(tiles[tile_iter], nchar(tiles[tile_iter])-10, nchar(tiles[tile_iter])), ".txt")
        )

        ### combine all dataframes, remove plotID except in first dataframe
        dat2 <- dat2[, -c(1)]
        dat3 <- dat3[, -c(1)]
        dat4 <- dat4[, -c(1)]
        dat <- cbind(dat1, dat2, dat3, dat4)

        ### prepare for acquiring the time series per pixel
        plots <- dat[, 1]
        dat <- as.data.frame(dat)
        data <- dat[, names(dat) != "plotID"]
        dat <- data

        for (iter in 1:nrow(dat)){
          print(paste0(iter, " out of ", nrow(dat)))
          df <- dat[iter, ]
          plot <- as.character(plots[iter])
          df <- reshape(df, direction="long", varying=colnames(df), sep="_")
          ### remove potential duplicates
          df <- df[!duplicated(as.character(substr(df$time, 1, 8))), ]

          ### drop id column
          df <- df[, !(colnames(df) %in% c("id"))]
          colnames(df)[2:length(colnames(df))] <- paste0(colnames(df)[2:length(colnames(df))], "_mean")

          df <- as.data.frame(df)
          colnames(df)[1] <- "date"
          df$plotID <- as.character(plot)
          df <- df %>% dplyr::select(plotID, date, everything())
          
          ### write the final table to disk
          write.table(df,
                      paste0(output_path,
                             df$plotID[1], '.csv'),
                      sep = ",", row.names = FALSE)

          ### end of for (iter in 1:nrow(dat)){
          ### iterating over large dataframe and saving the final time series
        }

        ### end of if (nrow(zones) > 0){
      }

      ### end of ### if more than 0 do
    }
  
 
  ### end of future function 
}


### define global variables
DATASET <- ''
REGION_FULL <- 'Germany'
REGION <- 'ger'
### path to FORCE tiles
PATH_TILE <- 'path/to/force/tiles'
### output path for resulting time series
OUTPUT_PATH <- paste0("path/to/result/folder/")
### path to meta file
ZONES_FILE <- 'path/to/metafile.gpkg'
### define end date of the time series
### startdate is not necessary, because following script will define a startdate
### here, we acquire all observations from first possible date (2015-07-01, start of S2 data)
### and the end date
ENDDATE <- 20221017

### prepare folder structure
dir.create(OUTPUT_PATH, recursive = TRUE)
dir.create(paste0(OUTPUT_PATH, 'meta/'))
dir.create(paste0(OUTPUT_PATH, 'tmp/'))

# define band designators
bands <- c('BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2') # 10 bands

### read shapefile
zones <- vect(ZONES_FILE, crs = 'epsg:3035')

### get tiles
tiles <- list.dirs(PATH_TILE)
# exclude parent directory
tiles <- tiles[2:length(tiles)]

### write metadata
zones$date <- as.character(zones$date)
writeVector(zones, paste0(OUTPUT_PATH, 'meta/', DATASET, '_', tolower(REGION_FULL), '.gpkg'), overwrite = TRUE)

### start parallel task
plan(multicore, workers = 15) 
future_map(1:length(tiles), function(tile_iter){get_ts_for_tile(tile_iter, tiles, zones, OUTPUT_PATH, ENDDATE)})
print("parallel task finished")

# stop the future plan
plan('sequential')