'''
output_spatial_predictions.py

Python code to output spatial predictions of melt pond as shapefiles to assess differences between different initialization states 

Requires QGIS Python API installed.

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
import richdem as rd
from rasterio import features
import os
import matplotlib as mpl
%matplotlib osx
from shapely.geometry import Point, Polygon
import rasterstats
import shapely
from pondtopo_functions import *

from qgis.core import *
QgsApplication.setPrefixPath("/Applications/QGIS.app/Contents/MacOS", True)
os.environ["PROJ_LIB"]="/Applications/QGIS-LTR.app/Contents/Resources/proj"
qgs = QgsApplication([], False)
qgs.initQgis()
from processing.core.Processing import Processing
Processing.initialize()
import processing


### Data configuration

data_path='../../../data_publish/'

# Pre-melt sea ice surface topography
demlist=[data_path+'ALS_DEM/20200116_01_ALS_DEM_0.5m_leg4CO_shift_pr_3413.tif',
    data_path+'Photo_DEM/20200423_DEM_corrected_crop_pr_3413.tif'
]

# Vector data defining study areas
floe_file = data_path+'Floe_contour.shp'

# Pond reference data, PASTA-ice vector output
classfile = data_path+'Reference_ponds/cummulative_pond_coverage.shp'


# if True, predictions are only calculated if processed files are not found
keep_existing=False

### Init variables

out_stats={}


### Read spatial data

floe_shape=gp.read_file(floe_file).to_crs(3413)
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
    
    # Save file, since some models read data themselves
    rxr_dem.dem_init.astype('float32').rio.to_raster(demfile[:-4]+'_clip.tif', driver="GTiff", compress="LZW")
    
    # flood pre-melt topography with different models
    rxr_dem['flooded_pysheds'] = (('y','x'),pysheds_predict(demfile,keep_existing))
    
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
    
    ### Stats retrieval, compare to different observed pond covers

    pond_df = gp.read_file(classfile).to_crs(3413)
    rxr_dem['class_map'] = (('y','x'), np.uint8(rasterize(pond_df, 'n_class', rxr_dem.dem_init[0,:,:], 'uint8', 255)))
        
    # use classified pond and open water pixels as true reference for pond predictions, inside the floe area, open water pixels are probably falsenegative pond pixels or melted-through ponds
    clas=np.logical_or(rxr_dem['class_map']==0, rxr_dem['class_map']==2) # pond reference raster
    pond_df = pond_df[np.logical_or(pond_df.n_class==0, pond_df.n_class==2)] # pond reference vector
    # clean pond reference vector
    class_df = pond_df.dissolve().explode(index_parts=True)
    class_df.reset_index(drop=True, inplace=True)
    
    # Flooded_DEM - DEM = Bathymetry
    bathy = (rxr_dem['flooded_pysheds']-rxr_dem.dem_init).astype('float32')
    
    # True, where ponds are predicted / in delineated basins
    hydr = (rxr_dem['flooded_pysheds']-rxr_dem.dem_init)>0. 
    
    # only consider objects>100 pixels
    hydr.values[:,:,0] = features.sieve(np.uint8(hydr.values[:,:,0]),100)
    
    # fill non-pond areas with fill_value and save
    bathy.values[:,:,0][~hydr.values[:,:,0]] = -32767
    bathy[:,:,0].rio.to_raster(demfile[:-4] + '_pysheds'+'_predicted_pond_bathy.tif', driver="GTiff", compress="LZW")
    
    # vectorize prediction if recalculation was desidered or file cannot be found
    if os.path.isfile(demfile[:-4] + '_pysheds'+'_flooded_dem.shp') and keep_existing:
        hydr_df = gp.read_file(demfile[:-4] + '_pysheds'+'_flooded_dem.shp')
    else:
        hydr_df = vectorize(np.uint8(hydr[:,:,0].values), mask, rxr_dem['flooded_pysheds'].rio.transform(), rxr_dem['flooded_pysheds'].rio.crs)
        hydr_df = hydr_df[hydr_df.pond==1]
        hydr_df.reset_index(drop=True, inplace=True)
        hydr_df.to_file(demfile[:-4] + '_pysheds'+'_flooded_dem.shp')
    
    ### pond depth, mean and max in polygons
    stats_ponds = rasterstats.zonal_stats(demfile[:-4] + '_pysheds'+'_flooded_dem.shp',demfile[:-4] + '_pysheds'+'_predicted_pond_bathy.tif',stats=["mean", "max"])
    
    for i,dummy in hydr_df.iterrows():
        hydr_df.loc[i,'mean_depth']=stats_ponds[i]['mean']
        hydr_df.loc[i,'max_depth']=stats_ponds[i]['max']
    
    ###########################
    # score areal predictions #
    ###########################
                
    min_depth = 0
        
    # areal predictions
    
    rxr_dem['hydr'] = (('y','x'),features.sieve(np.uint8(np.logical_and(hydr[:,:,0], bathy[:,:,0]>min_depth)),100))            
    
    ### location predictions, using PIA
    ### center point by PIA

    processing.run("native:poleofinaccessibility", 
        dict(
            INPUT=demfile[:-4] + '_pysheds'+'_flooded_dem.shp', 
            TOLERANCE=0.5, 
            OUTPUT=demfile[:-4] + '_pysheds'+'_flooded_dem_PIA.shp'))
    hydr_PIA = gp.read_file(demfile[:-4] + '_pysheds'+'_flooded_dem_PIA.shp')

    ### calculate PIA for clas

    processing.run("native:poleofinaccessibility", 
        dict(
            INPUT=classfile[:-4]+'_ponds_ow_in_floe.shp', 
            TOLERANCE=0.5, 
            OUTPUT=classfile[:-4]+'_ponds_ow_in_floe_PIA.shp'))
    class_PIA = gp.read_file(classfile[:-4]+'_ponds_ow_in_floe_PIA.shp')

    # center by deepest point (search for deepest point in each predicted pond, if multiple points exist, choose the one closest to the pond center)
    
    count_multiple_max=0
    print('search for deepest points, total number of iterations: '+str(len(hydr_df)))
    for ind, row in tqdm.tqdm(hydr_df.iterrows()):
        pond_bathy = bathy[:,:,0].rio.clip([row.geometry])
        y,x = np.where(pond_bathy == pond_bathy.max())
        if len(y)>1:
            count_multiple_max+=1
            distance=10000000.
            for (x,y) in zip(x,y):
                if np.sqrt((x-hydr_PIA.loc[ind,'geometry'].x)**2+(y-hydr_PIA.loc[ind,'geometry'].y)**2) < distance:
                    x_out=x
                    y_out=y
        else:
            x_out=x
            y_out=y
        hydr_df.loc[ind, 'deepest_point'] = Point(pond_bathy.x[x_out].values, pond_bathy.y[y_out].values)

    hydr_df_spatial_stats = hydr_df.copy()
    hydr_df_spatial_stats['prec_TP_PIA'] = 0
    hydr_df_spatial_stats['prec_TP_deepest'] = 0
    hydr_df_spatial_stats['dist_pole'] = hydr_PIA.dist_pole
    hydr_df_spatial_stats['PIA'] = hydr_PIA.geometry
    hydr_df_spatial_stats.loc[gp.sjoin(hydr_PIA, class_df, how='inner',predicate='within').index, 'prec_TP_PIA'] = 1 # 1 if pole of inaccessibility (PIA) of predicted pond is located in formed pond
    hydr_df_spatial_stats.loc[gp.sjoin(gp.GeoDataFrame({'geometry':hydr_df.deepest_point}).set_crs(3413), class_df, how='inner',predicate='within').index, 'prec_TP_deepest'] = 1  # 1 if deepest point of predicted pond is located in formed pond
    
    

    (hydr_df_spatial_stats[['prec_TP_PIA','prec_TP_deepest', 'dist_pole', 'geometry']]).to_file(demfile[:-4]+'_prec_TP_0cm_cumulative_pond.shp')
    gp.GeoDataFrame({
        'prec_TP_PIA':hydr_df_spatial_stats['prec_TP_PIA'],
        'prec_TP_deepest':hydr_df_spatial_stats['prec_TP_deepest'],
        'dist_pole':hydr_df_spatial_stats['dist_pole'],
        'geometry':hydr_df_spatial_stats['PIA']
    }, crs="EPSG:3413").to_file(demfile[:-4]+'_prec_TP_0cm_cumulative_PIA.shp')
    gp.GeoDataFrame({
        'prec_TP_PIA':hydr_df_spatial_stats['prec_TP_PIA'],
        'prec_TP_deepest':hydr_df_spatial_stats['prec_TP_deepest'],
        'dist_pole':hydr_df_spatial_stats['dist_pole'],
        'geometry':hydr_df_spatial_stats['deepest_point']
    }, crs="EPSG:3413").to_file(demfile[:-4]+'_prec_TP_0cm_cumulative_deepest_point.shp')
