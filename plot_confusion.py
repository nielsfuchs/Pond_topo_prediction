'''
plot_confusion.py

Python code to predict melt ponds from pre-melt sea ice surface topography, compare it to different reference pond data and plot confusion matrices for different models and scores. 

!Module Pondtopo_Functions requires Whiteboxtool Python Frontend folder WBT in the same directory 

Author: Niels Fuchs (2024), niels.fuchs@uni-hamburg.de, Github: https://github.com/nielsfuchs

Date: 2024-12-16
'''

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import rasterio as rio
import rioxarray as rxr
import geopandas as gp
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

# Pre-melt sea ice surface topography, ALS files originally from https://doi.org/10.1594/PANGAEA.950896, Photo+ALS files from https://doi.pangaea.de/10.1594/PANGAEA.949433
demlist=[
    data_path+'Photo_DEM/20200320_DEM_corrected_crop_pr_3413.tif',
    data_path+'Photo_DEM/20200423_DEM_corrected_crop_pr_3413.tif',
    data_path+'Photo_DEM/20200510_DEM_corrected_pr_3413.tif',
    data_path+'ALS_DEM/20200107_01_ALS_DEM_0.5m_leg4CO_shift_pr_3413.tif',
    data_path+'ALS_DEM/20200116_01_ALS_DEM_0.5m_leg4CO_shift_pr_3413.tif',
    data_path+'ALS_DEM/20200321_01_ALS_DEM_0.5m_leg4CO_shift_pr_3413.tif',
    data_path+'ALS_DEM/20200423_01_ALS_DEM_0.5m_leg4CO_shift_pr_3413.tif',
    data_path+'PhotoALS_DEM/20200321_01_DEM_int_PS_crop_0.5m_shift_crop_to_shape_pr_3413.tif',
    data_path+'PhotoALS_DEM/20200423_01_DEM_int_PS_crop_0.5m_shift_crop_to_shape_pr_3413.tif',
    data_path+'PhotoALS_DEM/20200510_01_DEM_int_PS_crop_0.5m_shift_crop_to_shape_pr_3413.tif'
]

# Vector data defining study areas
floe_file = data_path+'Floe_contour.shp'

# Pond reference data, PASTA-ice vector output, 30 June and 22 July classification from: https://doi.pangaea.de/10.1594/PANGAEA.949167
classlist=[
    data_path+'Reference_ponds/cumulative_pond_coverage.shp',
    data_path+'Reference_ponds/20200630_Ortho_UTM31N_5dm_main_surface_type_polygons.shp',
    data_path+'Reference_ponds/20200722_Ortho_UTM31N_5dm_main_surface_type_polygons_floe_pos_20200630.shp'
]

# if True, predictions are only calculated if processed files are not found
keep_existing=False

# output reference data as raster/GeoTiff
reference_raster_out=False

### Init variables

precision=np.zeros((len(demlist),3,3,11,3))
recall=np.zeros((len(demlist),3,3,11,3))
accuracy=np.zeros((len(demlist),3,3,11,2))
jaccard=np.zeros((len(demlist),3,3,11,2))
cohen=np.zeros((len(demlist),3,3,11,2))

mpf=np.zeros((len(demlist), 6, 11,2)) # [init_dem, [cumulative_pond_ref, pre-drainage_pond_ref, post-drainage_pond_ref, pysheds, richdem, whitebox], minimum depth, [mean_pond_depth, local_pond_depth]]
date=[]
out_stats={}


### Read spatial data

floe_shape=gp.read_file(floe_file).to_crs(3413)
excl_Jan=floe_shape[floe_shape.NAME=='Jan07']
excl_April=floe_shape[floe_shape.NAME=='April23']
floe_shape=floe_shape[floe_shape.NAME=='2']


for n, demfile in tqdm.tqdm(enumerate(demlist)):
    
    ### Model predictions part
    
    # load dem and instantly reduce file size
    rxr_dem = (rxr.open_rasterio(demfile).to_dataset(name='dem_init')).rio.clip_box(
    minx=floe_shape.bounds.values[0][0],
    miny=floe_shape.bounds.values[0][1],
    maxx=floe_shape.bounds.values[0][2],
    maxy=floe_shape.bounds.values[0][3],
    crs=floe_shape.crs)
    
    # downsample to GSD=0.5m
    rxr_dem = rxr_dem.rio.reproject(dst_crs=rxr_dem.rio.crs, resolution=0.5)
    
    ###Photo+ALS rasters need hole filling
    if n>6:
        attributes=rxr_dem.dem_init.attrs
        rxr_dem['dem_init'] = (('band','y','x'),rxr_dem.dem_init.rio.interpolate_na(method='linear').data)
        rxr_dem.dem_init.attrs = attributes
    
    # Save file, since some models read data themselves
    rxr_dem.dem_init.astype('float32').rio.to_raster(demfile[:-4]+'_clip.tif', driver="GTiff", compress="LZW")
    
    # flood pre-melt topography with different models
    rxr_dem['flooded_pysheds'] = (('y','x'),pysheds_predict(demfile,keep_existing))
    rxr_dem['flooded_richdem'] = (('y','x'),richdem_predict(demfile,keep_existing))
    rxr_dem['flooded_whitebox'] = (('y','x'),whitebox_predict(demfile,keep_existing))
    
    # clean up
    os.system('rm '+demfile[:-4]+'_clip.tif')
    
    ### Pre-processing for analysis
    
    # only consider MOSAiC CO2/leg4 summer floe area
    rxr_dem=rxr_dem.rio.clip(floe_shape.geometry)
    
    # define mask, True/1 inside floe area
    if np.isnan(rxr_dem.dem_init.attrs['_FillValue']):
        mask = np.isfinite(rxr_dem.dem_init)[0,:,:].values
    else:
        mask = (rxr_dem.dem_init!=rxr_dem.dem_init.attrs['_FillValue'])[0,:,:].values
    
    # denominators for spatial fractions
    n_valid_pixels = np.sum(mask)
    valid_area = float(n_valid_pixels*rxr_dem.dem_init.rio.transform()[0]**2)
    n_nan_pixels = n_valid_pixels-(len(rxr_dem.x)*len(rxr_dem.y))
    
    # plot labels (! only valid with the given DEMlist)
    if n<3:
        if n==0:
            date.append('20200321\nphoto')
        else:
            date.append((demfile.rsplit('/',1)[1]).split('_')[0]+'\nphoto')
    elif n>2 and n<7:
        date.append((demfile.rsplit('/',1)[1]).split('_')[0]+'\nALS')
    else:
        date.append((demfile.rsplit('/',1)[1]).split('_')[0]+'\nphoto+ALS')
        
    
    ### Stats retrieval, compare to different observed pond covers
    
    for m, classfile in enumerate(classlist):

        pond_df = gp.read_file(classfile).to_crs(3413)
        rxr_dem['class_map'] = (('y','x'), np.uint8(rasterize(pond_df, 'n_class', rxr_dem.dem_init[0,:,:], 'uint8', 255)))
            
        if reference_raster_out:
            out = np.uint8(np.zeros(rxr_dem.dem_init[0,:,:].shape))
            out[:,:] = 255
            shapes = ((geom,values) for geom,values in tqdm.tqdm(zip(pond_df.geometry,pond_df.n_class)))
            rxr_dem['class_map'] = (('y','x'),np.uint8(features.rasterize(shapes=shapes, fill=255, out=out, transform=rxr_dem.rio.transform())))
            rxr_dem.class_map.astype('uint8').rio.to_raster(classfile.rsplit('.',1)[0]+'.tif', driver="GTiff", compress="LZW")
       
        # use classified pond and open water pixels as true reference for pond predictions, inside the floe area, open water pixels are probably falsenegative pond pixels or melted-through ponds
        clas=np.logical_or(rxr_dem['class_map']==0, rxr_dem['class_map']==2) # pond reference raster
        pond_df = pond_df[np.logical_or(pond_df.n_class==0, pond_df.n_class==2)] # pond reference vector
        # clean pond reference vector
        class_df = pond_df.dissolve().explode(index_parts=True)
        class_df.reset_index(drop=True, inplace=True)
        class_df.to_file(classfile[:-4]+'_ponds_ow_in_floe.shp')
        
        for l, prediction in enumerate(['pysheds', 'richdem', 'whitebox']):
            
            # Flooded_DEM - DEM = Bathymetry
            bathy = (rxr_dem['flooded_'+prediction]-rxr_dem.dem_init).astype('float32')
            
            # True, where ponds are predicted / in delineated basins
            hydr = (rxr_dem['flooded_'+prediction]-rxr_dem.dem_init)>0. 
            
            # only consider objects>100 pixels
            hydr.values[:,:,0] = features.sieve(np.uint8(hydr.values[:,:,0]),100)
            
            # fill non-pond areas with fill_value and save
            bathy.values[:,:,0][~hydr.values[:,:,0]] = -32767
            bathy[:,:,0].rio.to_raster(demfile[:-4] + '_'+prediction+'_predicted_pond_bathy.tif', driver="GTiff", compress="LZW")
            
            # vectorize prediction if recalculation was desidered or file cannot be found
            if os.path.isfile(demfile[:-4] + '_'+prediction+'_flooded_dem.shp') and keep_existing:
                hydr_df = gp.read_file(demfile[:-4] + '_'+prediction+'_flooded_dem.shp')
            else:
                hydr_df = vectorize(np.uint8(hydr[:,:,0].values), mask, rxr_dem['flooded_'+prediction].rio.transform(), rxr_dem['flooded_'+prediction].rio.crs)
                hydr_df = hydr_df[hydr_df.pond==1]
                hydr_df.reset_index(drop=True, inplace=True)
                hydr_df.to_file(demfile[:-4] + '_'+prediction+'_flooded_dem.shp')
            
            ### pond depth, mean and max in polygons
            stats_ponds = rasterstats.zonal_stats(demfile[:-4] + '_'+prediction+'_flooded_dem.shp',demfile[:-4] + '_'+prediction+'_predicted_pond_bathy.tif',stats=["mean", "max"])
            
            for i,dummy in hydr_df.iterrows():
                hydr_df.loc[i,'mean_depth']=stats_ponds[i]['mean']
                hydr_df.loc[i,'max_depth']=stats_ponds[i]['max']
            
            ###########################
            # score areal predictions #
            ###########################
            
            ## by mean pond depth greater than 0m≤x≤0.20m
            
            for md, min_depth in enumerate(np.arange(0,0.22,0.02)):
            
                # MPF 
                mpf[n,3+l,md,0] = float(np.sum(hydr_df[hydr_df.mean_depth>min_depth].area)) / valid_area # predicted mpf
                mpf[n,m,md,0] = float(np.sum(class_df.area))/valid_area # observed mpf
            
                # areal predictions
                
                rxr_dem['hydr_md'] = (('y','x'),rasterize(hydr_df[hydr_df.mean_depth>min_depth], 'pond', rxr_dem.dem_init[0,:,:], 'uint8', 255) == 1)
                precision[n,m,l,md,0], recall[n,m,l,md,0], accuracy[n,m,l,md,0], jaccard[n,m,l,md,0], cohen[n,m,l,md,0] = get_stats(rxr_dem['hydr_md'], clas, mask)

            # by local depth greater than 0m≤x≤0.20m
            
            for ld, min_depth in enumerate(np.arange(0,0.22,0.02)):
                
                # MPF 
                
                mpf[n,2+l,ld,1] = float(np.sum(features.sieve(np.uint8(np.logical_and(hydr[:,:,0], bathy[:,:,0]>min_depth)),100))) / float(n_valid_pixels)    # predicted mpf
                mpf[n,m,ld,1] = float(np.sum(class_df.area))/valid_area # observed mpf
            
                # areal predictions
                
                rxr_dem['hydr_ld'] = (('y','x'),features.sieve(np.uint8(np.logical_and(hydr[:,:,0], bathy[:,:,0]>min_depth)),100))
                precision[n,m,l,ld,1], recall[n,m,l,ld,1], accuracy[n,m,l,ld,1], jaccard[n,m,l,ld,1], cohen[n,m,l,ld,1] = get_stats(rxr_dem['hydr_ld'], clas, mask)
                
                if n==1 and l==0 and m==0 and ld==0:
                    # output stats raster for plotting
                    rxr_output = rxr.raster_dataset.xarray.full_like(rxr_dem.dem_init[0,:,:], fill_value=255, dtype='uint8')
                    rxr_output *= ~mask
                    rxr_output += np.logical_and(clas, rxr_dem['hydr_ld']) * np.uint8(1) # TP
                    rxr_output += np.logical_and(rxr_dem['hydr_ld'], clas==False) * np.uint8(2) # FP
                    rxr_output += np.logical_and(clas, rxr_dem['hydr_ld']==False) * np.uint8(3) # FN
                    rxr_output.attrs.update({'_FillValue':255})
                    rxr_output.rio.to_raster(demfile[:-4] + '_'+prediction+'_stats_mindepth_'+str(min_depth*100)+'cm_cumulative.tif')
            

F_value = 2*precision*recall/(precision+recall)

######## plot V_II different scores

fig, ax = plt.subplots(3,1, sharex=True, sharey=True, constrained_layout=True, figsize=(7.57,4.86))
sorting=[3,4,0,5,7,1,6,8,2,9]

ax[0], c = plot_confusion(ax[0],accuracy[sorting,:,0,0,1].T)
ax[0].set_title('Accuracy')
ax[1], c = plot_confusion(ax[1],F_value[sorting,:,0,0,1].T)
ax[1].set_title('F-score')
ax[2], c = plot_confusion(ax[2],cohen[sorting,:,0,0,1].T)
ax[2].set_title(r'Cohen $\kappa$')

ax[-1].set_xticks(range(precision.shape[0]))
ax[-1].set_xticklabels(np.array(date)[sorting], rotation=80)

ax[-1].set_xlabel('Initialization DEM', fontsize=14)
ax[1].set_ylabel('Reference ponds', fontsize=14)

for i in range(3):
    ax[i].set_yticks(range(3))
    ax[i].set_yticklabels(['cumulative', 'pre-drainage', 'post-drainage'])
#    ax[i].tick_params('both', labelsize=)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.7])
cb=fig.colorbar(c, cax=cbar_ax)
cb.ax.tick_params(labelsize=12)
#cb.set_label(r'F-score', size=14)


######## plot V_II supplement with F-score

fig, ax = plt.subplots(3,1, sharex=True, sharey=True, constrained_layout=True, figsize=(7.57,4.86))
sorting=[3,4,0,5,7,1,6,8,2,9]
for l, prediction in enumerate(['Pysheds', 'RichDEM', 'Whitebox']):
    ax[l], c = plot_confusion(ax[l],F_value[sorting,:,l,0,1].T)
    #ax[l].imshow(F_value[:,:,l,3,1].T, cmap='PuBuGn', vmin=0, vmax=1)
    ax[l].set_title(prediction)

ax[-1].set_xticks(range(precision.shape[0]))
ax[-1].set_xticklabels(np.array(date)[sorting], rotation=80)

ax[-1].set_xlabel('Initialization DEM', fontsize=14)
ax[1].set_ylabel('Reference ponds', fontsize=14)

for i in range(3):
    ax[i].set_yticks(range(3))
    ax[i].set_yticklabels(['cumulative', 'pre-drainage', 'post-drainage'])
#    ax[i].tick_params('both', labelsize=)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.7])
cb=fig.colorbar(c, cax=cbar_ax)
cb.ax.tick_params(labelsize=12)
cb.set_label(r'F-score', size=14)



######## plot V_II supplement with cohen

fig, ax = plt.subplots(3,1, sharex=True, sharey=True, constrained_layout=True, figsize=(7.57,4.86))
sorting=[3,4,0,5,7,1,6,8,2,9]
for l, prediction in enumerate(['Pysheds', 'RichDEM', 'Whitebox']):
    ax[l], c = plot_confusion(ax[l],cohen[sorting,:,l,0,1].T)
    #ax[l].imshow(F_value[:,:,l,3,1].T, cmap='PuBuGn', vmin=0, vmax=1)
    ax[l].set_title(prediction)

ax[-1].set_xticks(range(cohen.shape[0]))
ax[-1].set_xticklabels(np.array(date)[sorting], rotation=80)

ax[-1].set_xlabel('Initialization DEM', fontsize=14)
ax[1].set_ylabel('Reference ponds', fontsize=14)

for i in range(3):
    ax[i].set_yticks(range(3))
    ax[i].set_yticklabels(['cumulative', 'pre-drainage', 'post-drainage'])
#    ax[i].tick_params('both', labelsize=)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.7])
cb=fig.colorbar(c, cax=cbar_ax)
cb.ax.tick_params(labelsize=12)
cb.set_label(r'Cohen-k coeffizient', size=14)


######## plot V_II supplement with accuracy

fig, ax = plt.subplots(3,1, sharex=True, sharey=True, constrained_layout=True, figsize=(7.57,4.86))
sorting=[3,4,0,5,7,1,6,8,2,9]
for l, prediction in enumerate(['Pysheds', 'RichDEM', 'Whitebox']):
    ax[l], c = plot_confusion(ax[l],accuracy[sorting,:,l,0,1].T)
    #ax[l].imshow(F_value[:,:,l,3,1].T, cmap='PuBuGn', vmin=0, vmax=1)
    ax[l].set_title(prediction)

ax[-1].set_xticks(range(accuracy.shape[0]))
ax[-1].set_xticklabels(np.array(date)[sorting], rotation=80)

ax[-1].set_xlabel('Initialization DEM', fontsize=14)
ax[1].set_ylabel('Reference ponds', fontsize=14)

for i in range(3):
    ax[i].set_yticks(range(3))
    ax[i].set_yticklabels(['cumulative', 'pre-drainage', 'post-drainage'])
#    ax[i].tick_params('both', labelsize=)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.7])
cb=fig.colorbar(c, cax=cbar_ax)
cb.ax.tick_params(labelsize=12)
cb.set_label(r'Accuracy', size=14)
