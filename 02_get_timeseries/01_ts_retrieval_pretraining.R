### this script takes as input a polygon metafile with the labels
### and outputs the area-weighted mean of those polygons
### over the overlapping pixels of S2
### in one time series csv file per sample

set.seed(123)

require(terra)
require(raster)
require(rgdal)
require(foreach)
library(lubridate)
require(stringr)
require(exactextractr)
require(sf)
require(dplyr)
require(data.table)

### define global variables
DATASET <- ''
REGION_FULL <- 'Germany'
### path to FORCE tiles
PATH_TILE <- 'path/to/force/tiles'
### output folder
OUTPUT_PATH <- ""
## path to meta file
ZONES_FILE <- 'path/to/metafile.gpkg'

### get tiles
tiles <- list.dirs(PATH_TILE)
# exclude parent directory
tiles <- tiles[2:length(tiles)]

### prepare folder structure
dir.create(OUTPUT_PATH)
dir.create(paste0(OUTPUT_PATH, 'meta/'))
dir.create('tmp_force_time_series_vrt/')

# define band designators
bands <- c('BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2') # 10 bands

statistics <- c('mean')

### read metadata
zones <- vect(ZONES_FILE, crs = 'epsg:3035')

### save metadata to disk
zones$date <- as.character(zones$date)
writeVector(zones, paste0(OUTPUT_PATH, 'meta/', DATASET, '.gpkg'), overwrite = TRUE)

### sf objec tneeded for exactextract function
zones <- st_as_sf(zones)

### initialize parallelization and foreach loop here
no_cpus <- 70

my.cluster <- parallel::makeCluster(
  no_cpus,
  type = "FORK", 
  outfile = ""
)

# register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)


### start of parallel loop
foreach(main_iterator = 1:length(tiles), .packages = "exactextractr") %dopar% {

  ### use only Sentinel-2 BOA data
  stck <- list.files(paste0(tiles[main_iterator]), pattern = ".SEN2(A|B)_BOA.tif$", full.names = TRUE)
  
  ### only if any SEN2A/SEN2B data exists in the tile
  if (length(stck) > 0){
    
    ### restrict to specific date
    years <- lapply(stck, substr, nchar(stck)-28, nchar(stck)-21)
    # Convert years to numeric
    years <- as.numeric(years)
    # Filter filenames based on the condition
    stck <- subset(stck, years <= 20221017)
    
    ### get extent of force tile
    xtnt <- as.polygons(ext(rast(stck[1])), crs = "epsg:3035")
    ### check which polygons are completely covered by tile extent
    rlt <- is.related(vect(zones), xtnt, "coveredby")
    
    ### extract those polygons
    zones_extract <- st_as_sf(zones[rlt, ])
    
    ### continue only if those polygons exist
    if (nrow(zones_extract) > 0){
      
      ### stack the datacube
      datacube <- rast(stck)
      
      ### rename bands
      names(datacube) <- rep(bands, length(names(datacube))/10)
      
      ### get date 
      date <- substr(stck, nchar(stck)-28, nchar(stck)-21)
      
      ### add date to datacube names in correct order
      names_vect = c()
      for (i in 1:length(date)){
        help_vect = paste0(bands, "_", date[i])
        names_vect = c(names_vect, help_vect)
      }
      names(datacube) <- names_vect
      
      ### iterate through the polygons
      for (zone_iterator in 1:nrow(zones_extract)){
        Sys.time()
        print("zone_iterator: ")
        print(zone_iterator)
        
        # initiate empty list
        lst <- vector(mode = "list", length = 0)
        
        ### initialize column names
        # initialize band columns in band_values object
        if ('quantile' %in% statistics){
          
          for (band in bands){
            
            # append all possible column names
            lst <- append(lst, paste0(statistics[-length(statistics)], '.', band))
            lst <- append(lst, paste0(ifelse(
              nchar(substr(as.character(quant), 3, 4)) == 1, 
              paste0('q', substr(as.character(quant), 3, 4), '0'), 
              paste0('q', substr(as.character(quant), 3, 4))), '.', band))
            
          }
          
          band_values <- unlist(lst)
          band_values <- setNames(data.frame(matrix(ncol = length(lst), nrow = length(date))), band_values)
          
        } else {
          
          for (band in bands){
            lst <- append(lst, paste0(bands, '_', statistics))
          }
          
          band_values <- unlist(lst)
          band_values <- setNames(data.frame(matrix(ncol = length(lst), nrow = length(date))), band_values)

        }

        ### add date to band_values
        band_values <- cbind(date, band_values)
        colnames(band_values)[1] <- 'date'
        
        ### execute the extraction
        band_df <- exact_extract(datacube, zones_extract[zone_iterator, ],
                                  fun = statistics, quantiles = quant,
                                  progress = FALSE)
        
        band_df <- reshape(band_df, direction="long", varying=colnames(band_df), sep="_")
        
        ### drop id column
        band_df <- band_df[, !(colnames(band_df) %in% c("id"))]
        colnames(band_df)[1] <- "date"
        
        ### add plotID
        band_df <- cbind(zones_extract$plotID[zone_iterator], band_df)
        colnames(band_df)[1] <- 'plotID'
        
        ### rename column names
        colnames(band_df)[3:length(colnames(band_df))] <- str_replace(colnames(band_df)[3:length(colnames(band_df))], "\\.", "_")
        
        ### export band values incl. date as csv file
        ### make sure that file name enables later identification to match labels
        write.table(band_df, 
                    paste0(OUTPUT_PATH, 
                            zones_extract$plotID[zone_iterator], '.csv'), 
                    sep = ",", row.names = FALSE)

        ### end of for-loop iterating over zones
      }
      
      
      ### end of if length(stck) > 0
    }
    
    ### end of if (nrow(zones_extract) > 0){
  }

  ### end of foreach loop
}


parallel::stopCluster(cl = my.cluster)

### next task: from the temp files, create time series csv files
### initialize parallelization and foreach loop here
no_cpus <- 25

my.cluster <- parallel::makeCluster(
  no_cpus,
  type = "FORK"
)

# register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)

### get list of temp files
df_list <- list.files('tmp_force_time_series_vrt', pattern = paste0(DATASET, '_', REGION, '.csv'), full.names = TRUE)

### get headers
headers <- fread(df_list[1], sep = ',', nrows = 0)
colnames(headers)

nrowfile <- fread(df_list[1], sep = ',')
chunks <- floor(seq.int(1, nrow(nrowfile), length.out = no_cpus))

### foreach loop is for the no_cpus chunks
foreach(main_iterator = 1:length(chunks)) %dopar% {
  
  rownumber = chunks[main_iterator + 1] - chunks[main_iterator]

  ### get big dataframe with only the revelant rows
  ### which enables the extraction of those time series
  big_df <- do.call(bind_rows, 
                    list(lapply(df_list, fread, 
                                sep = ',', nrows = rownumber, skip = chunks[main_iterator] - 1, 
                                col.names = colnames(headers), data.table = TRUE))
  )
  
  ### start for-loop iterating over each plotID
  ### (note that we have all observations for each plotID in the big_df dataframe)
  for (iter in unique(big_df$plotID)){
    plot <- big_df[big_df$plotID == iter, ]
    # move plotID to first position
    plot <- plot %>%
      dplyr::select(plotID, date, everything())
    setorder(plot, cols = "date")
    write.table(plot, 
                paste0(OUTPUT_PATH, iter, '.csv'), 
                sep = ",", row.names = FALSE)
    
    ### end of for (iter in unique(big_df$plotID)){
  }
  
  ### end of foreach loop iterating over row chunks
}
