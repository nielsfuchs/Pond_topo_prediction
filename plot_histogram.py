'''
plot_histogram.py

Python code to predict melt ponds from pre-melt sea ice surface topography, compare it to different reference pond data and plot line histograms of pond size distribution. 

Author: Niels Fuchs (2024), niels.fuchs@uni-hamburg.de, Github: https://github.com/nielsfuchs

Date: 2024-12-17
'''

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import rasterio as rio
import rioxarray as rxr
import geopandas as gp
from scipy import ndimage as ndi
import sys
from rasterio import features
import os
import matplotlib as mpl
from shapely.geometry import Point, Polygon
import rasterstats
import shapely
import time
from pondtopo_functions import *
%matplotlib osx



### Data configuration

data_path='[path to files]/' # data from Zenodo doi.org/10.5281/zenodo.14717210

# Pre-melt sea ice surface topography
demfile= data_path+'Photo_DEM/20200423_DEM_corrected_crop_pr_3413.tif'

# Vector data defining study areas
floe_file = data_path+'Floe_contour.shp'

# Pond reference data, PASTA-ice vector output. 30 June data from PANGAEA https://doi.pangaea.de/10.1594/PANGAEA.949167
classlist=[
    data_path+'Reference_ponds/cumulative_pond_coverage.shp',
    data_path+'Reference_ponds/20200630_Ortho_UTM31N_5dm_main_surface_type_polygons.shp'
]

# Areal averaged surface melt [m3 m-2] used in the semi-dynamic prediction
melt_range=0.026, 0.074

# output sieved to unsieved fraction of the prediction (result stated in the manuscript)
sieve_fraction = True

### Init variables

ref_dict={}
pred_dict={}

### Read spatial data

floe_shape=gp.read_file(floe_file).to_crs(3413)
floe_shape=floe_shape[floe_shape.NAME=='2']

# load dem and instantly reduce file size
rxr_dem = (rxr.open_rasterio(demfile).to_dataset(name='dem_init')).rio.clip_box(
minx=floe_shape.bounds.values[0][0],
miny=floe_shape.bounds.values[0][1],
maxx=floe_shape.bounds.values[0][2],
maxy=floe_shape.bounds.values[0][3],
crs=floe_shape.crs)

# downsample to GSD=0.5m
rxr_dem = rxr_dem.rio.reproject(dst_crs=rxr_dem.rio.crs, resolution=0.5)

# Save file, since pysheds uses file as input

rxr_dem.dem_init.astype('float32').rio.to_raster(demfile[:-4]+'_clip.tif', driver="GTiff", compress="LZW")

### Pysheds routine

grid = Grid.from_raster(demfile[:-4]+'_clip.tif')
dem = grid.read_raster(demfile[:-4]+'_clip.tif')

# Fill pits in DEM
pit_filled_dem = grid.fill_pits(dem.copy())

# Fill depressions in DEM
flooded_dem = grid.fill_depressions(pit_filled_dem.copy())

## Determine D8 flow directions from DEM

# Specify directional mapping
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

# Compute flow directions
fdir = grid.flowdir(pit_filled_dem.copy(), dirmap=dirmap)

# Calculate flow accumulation
# --------------------------
acc = grid.accumulation(fdir.copy(), dirmap=dirmap)

#############################################


########## newly developped accumulation prediction part

### find accumulation endpoints (every pixel in the accumulation map is the sum of all upstream accumulations. Hence, local maxima are the endpoints of accumulation streams)

coords = np.stack((np.where(ndi.maximum_filter(np.uint16(acc), size=3)==np.uint16(acc)))).T
max_acc = np.zeros(acc.shape)
max_acc[coords[:,0],coords[:,1]] = acc[coords[:,0],coords[:,1]]

### bathymetry

depth=flooded_dem-dem

# add output to riox dataset and clip to floe shape

rxr_dem['depth'] = (('y','x'),depth)
rxr_dem['max_acc'] = (('y','x'),max_acc)
rxr_dem=rxr_dem.rio.clip(floe_shape.geometry)

# read clipped array for further processing

depth = rxr_dem['depth'].data
max_acc = rxr_dem['max_acc'].data

# define mask, True/1 inside floe area
if np.isnan(rxr_dem.dem_init.attrs['_FillValue']):
    mask = np.isfinite(rxr_dem.dem_init)[0,:,:].values
else:
    mask = (rxr_dem.dem_init!=rxr_dem.dem_init.attrs['_FillValue'])[0,:,:].values

# denominators for spatial fractions
n_valid_pixels = np.sum(mask)
valid_area = float(n_valid_pixels*rxr_dem.dem_init.rio.transform()[0]**2)
n_nan_pixels = n_valid_pixels-(len(rxr_dem.x)*len(rxr_dem.y))

### read reference pond size distributions
for n_c, classfile in enumerate(classlist):
    
    pond_df = gp.read_file(classfile).to_crs(3413)
    
    # use riox dataset as geotransform reference
    rxr_dem['class_map'] = (('y','x'), np.uint8(rasterize(pond_df, 'n_class', rxr_dem.dem_init[0,:,:], 'uint8', 255)))

    clas=np.logical_or(rxr_dem['class_map']==0, rxr_dem['class_map']==2)

    ref_dict[n_c] = vectorize(np.uint8(clas), mask, dem.affine, dem.crs)

########### semi-dynamic prediction

d=0. # minimum depth

depth_bool=depth>d

depth_bool=np.int64(depth_bool)

if sieve_fraction:
    print(np.sum(depth_bool), n_valid_pixels, np.sum(depth_bool)/n_valid_pixels, 'unsieved')

depth_bool = features.sieve(np.uint8(depth_bool), 100)

if sieve_fraction:
    print(np.sum(depth_bool), n_valid_pixels, np.sum(depth_bool)/n_valid_pixels, 'sieved')

labels=ndi.label(depth_bool)    # gives every area a unique label as raster map

pond_label = labels[0].copy()   

# calculate pond prediction map with different surface meltwater amounts

for n_d, surface_melt in tqdm.tqdm(enumerate(melt_range)): 

    vol=np.zeros(labels[1])
    ac=np.zeros(labels[1])
    overflow=np.zeros(labels[1])
    pond_label_new = pond_label.copy()
    pond_vol_bool = np.ones(pond_label_new.shape)*-32767

    # area and volume per pixel, assumes rectangular, cartesian grid
    area = grid.affine[0]*grid.affine[4]*-1
    volume = grid.affine[0]*grid.affine[4]*-1*surface_melt

    # volume and accumulation volume within ponds. 
    for i in range(1,labels[1]):
        vol[i] = np.sum(depth[pond_label==i])*area
        ac[i] = np.sum(max_acc[pond_label==i])*volume

    ### recalculate theoertical pond expansion based on available melt mater. Returns raster mal with 1 for filled pond areas, 2 for overflowing pond areas and -32767 where not enough melt water is available, 0 for non ponded areas
    for i in range(1,labels[1]):
        if vol[i]>ac[i]:
            level=np.max(depth[pond_label==i])
            v=0
            while v < ac[i]:
                level-=0.01 # filling steps of +1cm 
                v=np.sum(np.clip(depth[pond_label==i]-level,0,None))*area
            pond_label_new[np.logical_and(pond_label==i, depth<level)]=-32767
            pond_vol_bool[np.logical_and(pond_label==i, depth>=level)]=1
        else:
            pond_vol_bool[pond_label==i]=2

    pond_vol_bool[pond_label==0]=0 

    rxr_dem['predicted'] = (('y','x'),pond_vol_bool)
    
    pred_dict[n_d] = vectorize(np.uint8(np.isin(rxr_dem['predicted'].values,[1,2])), mask, dem.affine, dem.crs)
    
## plot results

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(7.57,4.86))

hist, bins = np.histogram(ref_dict[1][np.isin(ref_dict[1].pond, [1])].area, bins=np.hstack((np.array(0),np.arange(40,420,20),np.array(1000000))))
ax.plot(np.hstack((np.array(30), np.arange(50,410,20), np.array(410))), hist, color='teal', ls='-', label='pre-drainage observed') # plot vertices at bin centers
hist, bins = np.histogram(pred_dict[1][np.isin(pred_dict[1].pond, [1,2])].area, bins=np.hstack((np.array(0),np.arange(40,420,20),np.array(1000000))))
ax.plot(np.hstack((np.array(30), np.arange(50,410,20), np.array(410))), hist, color='skyblue', ls='--', label='pre-drainage predicted (surface melt=0.026m)')
hist, bins = np.histogram(ref_dict[0][np.isin(ref_dict[0].pond, [1])].area, bins=np.hstack((np.array(0),np.arange(40,420,20),np.array(1000000))))
ax.plot(np.hstack((np.array(30), np.arange(50,410,20), np.array(410))), hist, color='darkorange', ls='-.', label='cumulative observed')
hist, bins = np.histogram(pred_dict[0][np.isin(pred_dict[0].pond, [1,2])].area, bins=np.hstack((np.array(0),np.arange(40,420,20),np.array(1000000))))
ax.plot(np.hstack((np.array(30), np.arange(50,410,20), np.array(410))), hist, color='goldenrod', ls=':', label='cumulative predicted  (surface melt=0.074m)')
ax.set_yscale('log')
ax.set_xticks([30,100,200,300,410])
ax.set_xticklabels(['<50','100','200','300','≥400'])
ax.tick_params('both', labelsize=12)
ax.legend(loc='upper right', fontsize=13)
ax.set_xlabel(r'Pond size [m$^2$]'+'\n'+r'(bin width=20 m$^2$)', fontsize=13)
ax.set_ylabel(r'Number of ponds', fontsize=13)

