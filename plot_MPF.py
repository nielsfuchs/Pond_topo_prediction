'''
plot_MPF.py

Python code to predict melt ponds from pre-melt sea ice surface topography using static and semi-dynamic approach, and plot melt pond fraction (MPF)

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

# Pond reference data, PASTA-ice vector output
classfile = data_path+'Reference_ponds/cumulative_pond_coverage.shp'

# configuration

n_subsets = 10
subset_size = 10000
max_surface_melt = 0.15 # Smith et al approximated (15cm: snow + surface melt), snow=5cm

# initialize variables

melt_range=np.linspace(0,max_surface_melt,15)

mpf=np.zeros((1,n_subsets,len(melt_range),3))

### Read spatial data

floe_shape=gp.read_file(floe_file).to_crs(3413)
floe_shape=floe_shape[floe_shape.NAME=='2'] # at one point, the floe was stored as number 2 in that file ;)

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

# get n random bootstrap sample points

subset_list = []

for n_s in range(n_subsets):
    subset_list.append(sorted(np.random.choice(range(np.sum(mask)), subset_size, replace=False).tolist()))

# reference data

pond_df = gp.read_file(classfile).to_crs(3413)
rxr_dem['class_map'] = (('y','x'), np.uint8(rasterize(pond_df, 'n_class', rxr_dem.dem_init[0,:,:], 'uint8', 255)))
clas=np.logical_or(rxr_dem['class_map']==0, rxr_dem['class_map']==2) # use classified pond and open water pixels as true reference for pond predictions, inside the floe area, open water pixels are probably falsenegative pond pixels or melted-through ponds

########### semi-dynamic prediction

d=0. # minimum depth

depth_bool=depth>d

depth_bool=np.int64(depth_bool)

depth_bool = features.sieve(np.uint8(depth_bool), 100)

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
    

    if surface_melt==max_surface_melt:
        # output static prediction raster map 1:TP, 2:FP, 3:FN, 0:else 
        rxr_output = rxr.raster_dataset.xarray.full_like(rxr_dem.dem_init[0,:,:], fill_value=255, dtype='uint8')
        rxr_output *= ~mask
        rxr_output += np.logical_and(clas, np.isin(rxr_dem['predicted'],[-32767, 1, 2])) * np.uint8(1) # TP
        rxr_output += np.logical_and(np.isin(rxr_dem['predicted'],[-32767, 1, 2]), clas==False) * np.uint8(2) # FP
        rxr_output += np.logical_and(clas, rxr_dem['predicted']==0) * np.uint8(3) # FN
        rxr_output.attrs.update({'_FillValue':255})
        rxr_output.rio.to_raster(demfile[:-4] + '_pysheds_stats_static_prediction_cum_pond_coverage.tif')
        
        # output semi-dynamix raster map for total surface melt, 1:TP, 2:FP, 3:FN, 0:else
        rxr_output = rxr.raster_dataset.xarray.full_like(rxr_dem.dem_init[0,:,:], fill_value=255, dtype='uint8')
        rxr_output *= ~mask
        rxr_output += np.logical_and(clas, rxr_dem['predicted']>0) * np.uint8(1) # TP
        rxr_output += np.logical_and(rxr_dem['predicted']>0, clas==False) * np.uint8(2) # FP
        rxr_output += np.logical_and(clas, rxr_dem['predicted']<1) * np.uint8(3) # FN
        rxr_output.attrs.update({'_FillValue':255})
        rxr_output.rio.to_raster(demfile[:-4] + '_pysheds_stats_mindepth_'+str(d*100)+'cm_acc_'+str(surface_melt*100)+'cm_surf_melt_cumulative_coverage.tif')
        
        # output polygons that mark overflowing ponds
        vectorize(np.uint8(rxr_dem['predicted'].values==2), mask, dem.affine, dem.crs).to_file(demfile[:-4] + '_pysheds_stats_mindepth_'+str(d*100)+'cm_acc_spill_'+str(surface_melt*100)+'cm_surf_melt.shp')
       
    # derive statistics for subsets
    
    for n_s, subset in enumerate(subset_list):
        
        mpf[0,n_s,n_d,0] = np.sum(clas.values[np.where(mask)][subset])/subset_size # reference melt pond coverage
        mpf[0,n_s,n_d,1] = np.sum(np.isin(rxr_dem['predicted'].values[np.where(mask)][subset],[1,2]))/subset_size # semi-dynamic positive predictions
        mpf[0,n_s,n_d,2] = np.sum(np.isin(rxr_dem['predicted'].values[np.where(mask)][subset],[1,2,-32767]))/subset_size # static positive predictions
    
mpf*=100 # make percent

# plot

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(7.57,4.86))

ax.plot(melt_range, np.mean(mpf[0,:,:,2].T, axis=1), label='static prediction ± std', color='saddlebrown')
ax.fill_between(melt_range, np.mean(mpf[0,:,:,2].T, axis=1)-np.std(mpf[0,:,:,2].T, axis=1), np.mean(mpf[0,:,:,2].T, axis=1)+np.std(mpf[0,:,:,2].T, axis=1), alpha=0.3, color='saddlebrown')
ax.plot(melt_range, np.mean(mpf[0,:,:,1].T, axis=1), label='semi-dyn. prediction  ± std', color='sandybrown')
ax.fill_between(melt_range, np.mean(mpf[0,:,:,1].T, axis=1)-np.std(mpf[0,:,:,1].T, axis=1), np.mean(mpf[0,:,:,1].T, axis=1)+np.std(mpf[0,:,:,1].T, axis=1), alpha=0.3, color='sandybrown')

ax.axhline(22.4, color='black', ls='-', label='pre-drainage obs. ± confidence range')
ax.axhspan(21.5, 24.8, color='silver', alpha=0.3)
ax.axhline(37.2, color='black', ls='-.', label='cumulative obs.')

ax.vlines(0.026, 0, 22.4, color='grey', ls=':')
ax.vlines(0.074, 0, 37.2, color='grey', ls=':')

ax.set_yticks([0,10,22.4,30, 37.2, 40,48.99])
ax.set_xticks([0,0.026,0.05, 0.074, 0.1, 0.15])
ax.set_ylabel('Melt pond fraction MPF %', fontsize=14)
ax.set_xlabel(r'Surface melt [m]'+'\n'+' ', fontsize=14)
ax.tick_params('both', labelsize=12)
ax.legend(loc='lower right', fontsize=14)
ax.set_xlim([0,max_surface_melt])
ax.set_ylim([0,55])

