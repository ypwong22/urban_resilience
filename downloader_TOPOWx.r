# Dependent on the AOI and climateR packages
# https://github.com/mikejohnson51/climateR
library(AOI)
library(climateR)
library(sf)
library(raster)
library(dplyr) # Load dplyr package
setwd("/lustre/haven/proj/UTK0134/DATA/Meteorological/TOPOWx")

start.List = paste(seq(1980, 2016, 5), "-01-01", sep = "")
end.List = c(paste(seq(1984, 2015, 5), "-12-31", sep = ""), "2016-12-31")
name.List = c()
for (i in seq(1, length(start.List))){
  start.Y = as.numeric(strsplit(start.List[i], "-")[[1]][1])
  name.List = c(name.List, paste0(start.Y, "01", "01", "-",
                                  start.Y + 4, "12", "31"))
}

# Create a sf object from shapefile
# obj0 = st_read("./fid_shp/US_urbanCluster_500km_Union_dissolveboundaries_buffered.shp")
obj0 = st_read("/lustre/haven/proj/UTK0134/URBAN_LDRD/intermediate/urban_mask/city_boundary_3x_merged_buffer25km.shp")

for (fid in seq(0,84)){
# fid = 0

  if ((fid == 22) | (fid == 31)){
    start.List = paste(seq(1980, 2016, 1), "-01-01", sep = "")
    end.List = paste(seq(1980, 2016, 1), "-12-31", sep = "")
    name.List = c()
    for (i in seq(1, length(start.List))){
      start.Y = as.numeric(strsplit(start.List[i], "-")[[1]][1])
      name.List = c(name.List, paste0(start.Y, "01", "01", "-",
                                      start.Y, "12", "31"))
    }
  } else {
    start.List = paste(seq(1980, 2016, 5), "-01-01", sep = "")
    end.List = c(paste(seq(1984, 2015, 5), "-12-31", sep = ""), "2016-12-31")
    name.List = c()
    for (i in seq(1, length(start.List))){
      start.Y = as.numeric(strsplit(start.List[i], "-")[[1]][1])
      name.List = c(name.List, paste0(start.Y, "01", "01", "-",
                                      start.Y + 4, "12", "31"))
    }
  }

  for (i in seq(1,length(end.List))){

    # Select the FID
    obj = obj0 %>% filter(FID_1 == fid)

    # Create an AOI from the sf object
    aoi = aoi_get(x = obj)

    # Download the data
    # 10 seconds for fid 1
    system.time({
        p = getTopoWX(aoi, param = c('tmax','tmin'),
                      startDate = start.List[i], endDate = end.List[i])
    })

    # Stack the data
    r = raster::stack(p)

    ## Not working
    ### Read the correct spatial projection
    ##obj2 = raster(paste0("/lustre/haven/proj/UTK0134/SM_ECO/intermediate/gee_single/NLCD/NLCD_",
    ##                     sprintf(fid, fmt = "%02d"), ".tif"))

    ### Reproject the data
    ##r2 = projectRaster(r, crs = CRS(crs(obj2, asText = T)), method = "bilinear")
    ##r2 = crop(r2, extent(obj2))

    # Save the data to file
    writeRaster(r, paste0("fid", fid, "_", name.List[i], ".tif"), format="GTiff", overwrite = T)
  }
}