'''
plot_MPF_and_stats.py

Python code to predict melt ponds from pre-melt sea ice surface topography using static and semi-dynamic approach, compare it to cumulative reference pond data using bootstraping subsamples and plot predictive skills dependent on a minimum depth threshold

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

data_path='../../../data_publish/'

# Pre-melt sea ice surface topography
demfile= data_path+'Photo_DEM/20200423_DEM_corrected_crop_pr_3413.tif'

# Vector data defining study areas
floe_file = data_path+'Floe_contour.shp'

# Pond reference data, PASTA-ice vector output
classfile= data_path+'Reference_ponds/cumulative_pond_coverage.shp'


# configuration

n_subsets = 10
subset_size = 10000
surface_melt = 0.15 # Smith et al approximated (15cm: snow + surface melt), snow=5cm

# initialize variables

depth_range=np.arange(0,0.52,0.02)

precision=np.zeros((1,n_subsets,len(depth_range),4))
recall=np.zeros((1,n_subsets,len(depth_range),4))
accuracy=np.zeros((1,n_subsets,len(depth_range),4))
jaccard=np.zeros((1,n_subsets,len(depth_range),4))
cohen=np.zeros((1,n_subsets,len(depth_range),4))
mpf=np.zeros((1,n_subsets,len(depth_range),3))

date=[]


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

# get n random bootstrap sample points

subset_list = []

for n_s in range(n_subsets):
    subset_list.append(sorted(np.random.choice(range(np.sum(mask)), subset_size, replace=False).tolist()))

# reference data

pond_df = gp.read_file(classfile).to_crs(3413)
rxr_dem['class_map'] = (('y','x'), np.uint8(rasterize(pond_df, 'n_class', rxr_dem.dem_init[0,:,:], 'uint8', 255)))
clas=np.logical_or(rxr_dem['class_map']==0, rxr_dem['class_map']==2) # use classified pond and open water pixels as true reference for pond predictions, inside the floe area, open water pixels are probably falsenegative pond pixels or melted-through ponds

########### static and semi-dynamic prediction

for n_d, d in tqdm.tqdm(enumerate(depth_range)):

    d=0. # minimum depth

    depth_bool=depth>d

    depth_bool=np.int64(depth_bool)
    
    depth_bool = features.sieve(np.uint8(depth_bool), 100)

    labels=ndi.label(depth_bool)    # gives every area a unique label as raster map

    pond_label = labels[0].copy()   
    
    # semi-dynamic variables

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

    ### recalculate theoertical pond expansion based on available melt mater. Returns raster map with 1 for filled pond areas, 2 for overflowing pond areas and -32767 where not enough melt water is available, 0 for non ponded areas
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
    
    # derive statistics for subsets
     
    for n_s, subset in enumerate(subset_list):
        
        precision[0,n_s,n_d,0], recall[0,n_s,n_d,0], accuracy[0,n_s,n_d,0], jaccard[0,n_s,n_d,0], cohen[0,n_s,n_d,0] = get_stats(rxr_dem['predicted'].values[np.where(mask)][subset]==1, 
        clas.values[np.where(mask)][subset], np.ones((subset_size),dtype='bool')) # filled pond areas
        precision[0,n_s,n_d,1], recall[0,n_s,n_d,1], accuracy[0,n_s,n_d,1], jaccard[0,n_s,n_d,1], cohen[0,n_s,n_d,1] = get_stats(rxr_dem['predicted'].values[np.where(mask)][subset]==2, 
        clas.values[np.where(mask)][subset], np.ones((subset_size),dtype='bool')) # overflowing pond areas
        precision[0,n_s,n_d,2], recall[0,n_s,n_d,2], accuracy[0,n_s,n_d,2], jaccard[0,n_s,n_d,2], cohen[0,n_s,n_d,2] = get_stats(np.isin(rxr_dem['predicted'].values[np.where(mask)][subset],[1,2]), 
        clas.values[np.where(mask)][subset], np.ones((subset_size),dtype='bool')) # semi-dynamically positive predicted pond areas
        precision[0,n_s,n_d,3], recall[0,n_s,n_d,3], accuracy[0,n_s,n_d,3], jaccard[0,n_s,n_d,3], cohen[0,n_s,n_d,3] = get_stats(np.isin(rxr_dem['predicted'].values[np.where(mask)][subset],[1,2,-32767]), 
        clas.values[np.where(mask)][subset], np.ones((subset_size),dtype='bool')) # static positive predicted pond areas
        
        mpf[0,n_s,n_d,0] = np.sum(clas.values[np.where(mask)][subset])/subset_size # reference melt pond coverage
        mpf[0,n_s,n_d,1] = np.sum(np.isin(rxr_dem['predicted'].values[np.where(mask)][subset],[1,2]))/subset_size # semi-dynamic positive predictions
        mpf[0,n_s,n_d,2] = np.sum(np.isin(rxr_dem['predicted'].values[np.where(mask)][subset],[1,2,-32767]))/subset_size # static positive predictions
    
# derive F-score
F_value = 2*precision*recall/(precision+recall)
# make percent
mpf*=100

# plot

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(7.57,4.86))

#accuracy
ax.plot(depth_range, np.mean(accuracy[0,:,:,3].T, axis=1), label='Accuracy', color='chocolate')
ax.fill_between(depth_range, np.mean(accuracy[0,:,:,3].T, axis=1)-np.std(accuracy[0,:,:,3].T, axis=1), np.mean(accuracy[0,:,:,3].T, axis=1)+np.std(accuracy[0,:,:,3].T, axis=1), alpha=0.3, color='chocolate')

ax.plot(depth_range, np.mean(accuracy[0,:,:,2].T, axis=1), label='Accuracy', ls='--', color='sandybrown')
ax.fill_between(depth_range, np.mean(accuracy[0,:,:,2].T, axis=1)-np.std(accuracy[0,:,:,2].T, axis=1), np.mean(accuracy[0,:,:,2].T, axis=1)+np.std(accuracy[0,:,:,2].T, axis=1), alpha=0.3, color='sandybrown')


#precision
ax.plot(depth_range, np.mean(precision[0,:,:,3].T, axis=1), label='Precision', color='darkblue')
ax.fill_between(depth_range, np.mean(precision[0,:,:,3].T, axis=1)-np.std(precision[0,:,:,3].T, axis=1), np.mean(precision[0,:,:,3].T, axis=1)+np.std(precision[0,:,:,3].T, axis=1), alpha=0.3, color='darkblue')

ax.plot(depth_range, np.mean(precision[0,:,:,2].T, axis=1), label='Precision', ls='--', color='lightblue')
ax.fill_between(depth_range, np.mean(precision[0,:,:,2].T, axis=1)-np.std(precision[0,:,:,2].T, axis=1), np.mean(precision[0,:,:,2].T, axis=1)+np.std(precision[0,:,:,2].T, axis=1), alpha=0.3, color='lightblue')

#recall
ax.plot(depth_range, np.mean(recall[0,:,:,3].T, axis=1), label='Recall', color='darkorange')
ax.fill_between(depth_range, np.mean(recall[0,:,:,3].T, axis=1)-np.std(recall[0,:,:,3].T, axis=1), np.mean(recall[0,:,:,3].T, axis=1)+np.std(recall[0,:,:,3].T, axis=1), alpha=0.3, color='darkorange')

ax.plot(depth_range, np.mean(recall[0,:,:,2].T, axis=1), label='Recall', ls='--', color='orange')
ax.fill_between(depth_range, np.mean(recall[0,:,:,2].T, axis=1)-np.std(recall[0,:,:,2].T, axis=1), np.mean(recall[0,:,:,2].T, axis=1)+np.std(recall[0,:,:,2].T, axis=1), alpha=0.3, color='orange')

#F-score
ax.plot(depth_range, np.mean(F_value[0,:,:,3].T, axis=1), label='F-score', color='darkred')
ax.fill_between(depth_range, np.mean(F_value[0,:,:,3].T, axis=1)-np.std(F_value[0,:,:,3].T, axis=1), np.mean(F_value[0,:,:,3].T, axis=1)+np.std(F_value[0,:,:,3].T, axis=1), alpha=0.3, color='darkred')

ax.plot(depth_range, np.mean(F_value[0,:,:,2].T, axis=1), label='F-score', ls='--', color='red')
ax.fill_between(depth_range, np.mean(F_value[0,:,:,2].T, axis=1)-np.std(F_value[0,:,:,2].T, axis=1), np.mean(F_value[0,:,:,2].T, axis=1)+np.std(F_value[0,:,:,2].T, axis=1), alpha=0.3, color='red')

#Cohen
ax.plot(depth_range, np.mean(cohen[0,:,:,3].T, axis=1), label=r'Cohen $\kappa$', color='darkgreen')
ax.fill_between(depth_range, np.mean(cohen[0,:,:,3].T, axis=1)-np.std(cohen[0,:,:,3].T, axis=1), np.mean(cohen[0,:,:,3].T, axis=1)+np.std(cohen[0,:,:,3].T, axis=1), alpha=0.3, color='darkgreen')

ax.plot(depth_range, np.mean(cohen[0,:,:,2].T, axis=1), label=r'Cohen $\kappa$', ls='--', color='lightgreen')
ax.fill_between(depth_range, np.mean(cohen[0,:,:,2].T, axis=1)-np.std(cohen[0,:,:,2].T, axis=1), np.mean(cohen[0,:,:,2].T, axis=1)+np.std(cohen[0,:,:,2].T, axis=1), alpha=0.3, color='lightgreen')

#jaccard
ax.plot(depth_range, np.mean(jaccard[0,:,:,3].T, axis=1), label='Jaccard', color='purple')
ax.fill_between(depth_range, np.mean(jaccard[0,:,:,3].T, axis=1)-np.std(jaccard[0,:,:,3].T, axis=1), np.mean(jaccard[0,:,:,3].T, axis=1)+np.std(jaccard[0,:,:,3].T, axis=1), alpha=0.3, color='purple')

ax.plot(depth_range, np.mean(jaccard[0,:,:,2].T, axis=1), label='Jaccard', ls='--', color='orchid')
ax.fill_between(depth_range, np.mean(jaccard[0,:,:,2].T, axis=1)-np.std(jaccard[0,:,:,2].T, axis=1), np.mean(jaccard[0,:,:,2].T, axis=1)+np.std(jaccard[0,:,:,2].T, axis=1), alpha=0.3, color='orchid')

## dummy
ax.plot([],[], color='grey', label='static ± std')
ax.plot([],[], color='lightgrey', ls='--', label='semi-dyn. ± std')

ax.text(0.26,0.93,'Precision', color='darkblue',fontsize=13)
ax.text(0.4,0.68,'Accuracy', color='chocolate',fontsize=13)
ax.text(0.4,0.18,'F-Score', color='darkred',fontsize=13)
ax.text(0,0.34,r'Cohen $\kappa$', color='darkgreen',fontsize=13)
ax.text(0.24,0.07,'Jaccard', color='purple',fontsize=13)
ax.text(0,0.81,'Recall', color='darkorange',fontsize=13)

# legend
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])

handles, labels = ax.get_legend_handles_labels()
legend_sorting = [-2,-1]#, 1,3,5,7,9,11]
ax.legend(np.array(handles)[legend_sorting], np.array(labels)[legend_sorting], loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=12, ncol=2)#4)

# misc
ax.set_xlabel(r'Minimum depth $d$ [m]', fontsize=14)
ax.set_ylabel('Score', fontsize=14)
ax.tick_params('both', labelsize=10)
