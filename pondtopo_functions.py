'''
plottopo_functions.py

Python module containing different functions used in a study that compares pre-melt sea ice topography with the summer melt pond coverage

Requires WhiteBoxTools Extensions folder WBT in the same directory

Author: Niels Fuchs (2024), niels.fuchs@uni-hamburg.de, Github: https://github.com/nielsfuchs

Date: 2024-12-16
'''

from rasterio import features
import geopandas as gp
from pysheds.grid import Grid
import richdem as rd
from WBT.whitebox_tools import WhiteboxTools
from sklearn.metrics import cohen_kappa_score
import os
import numpy as np
import time

def vectorize(raster, mask, transform, crs):
    results = (
                {'properties': {'pond': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    features.shapes(raster, mask=mask, transform=transform)))
    geoms = list(results)
    gpd  = gp.GeoDataFrame.from_features(geoms)
    gpd.crs=crs
    return gpd
    
def rasterize(df, column, ref_rxr_raster, dtype, nan):
    out = np.zeros(ref_rxr_raster.shape).astype(dtype)
    out[:,:] = nan
    shapes = ((geom,values) for geom,values in zip(df.geometry,df[column]))
    return features.rasterize(shapes=shapes, fill=nan, out=out, transform=ref_rxr_raster.rio.transform())
    
    
def get_stats(predictor, target, mask):
    ### mask=1 inside valid area, =0 outside of valid area
    TP=float(np.sum(np.logical_and(target, predictor)))
    FP=float(np.sum(np.logical_and(predictor, target==False)))
    FN=float(np.sum(np.logical_and(target, predictor==False)))
    TN=float(np.sum(np.logical_and(target==False, predictor==False))-np.sum(~mask))   # subtract all pixels outside of clipped area
    precision=TP/np.float64(TP+FP)
    recall=TP/np.float64(TP+FN)
    accuracy=(TP+TN)/np.float64(TP+FP+FN+TN)
    jaccard = TP / np.float64(TP+FP+FN)
    if np.any(mask!=1):
        # mask data only when it still contains invalid area, if masked before, dtypes often don't have .values attribute
        cohen = cohen_kappa_score(target.values[mask], predictor.values[mask])
    else:
        cohen = cohen_kappa_score(target, predictor)
    return precision, recall, accuracy, jaccard, cohen
    


def pysheds_predict(demfile, keepexist):
    
    if os.path.isfile(demfile[:-4] + '_pysheds_flooded_dem.tif') and keepexist:
        grid = Grid.from_raster(demfile[:-4] + '_pysheds_flooded_dem.tif')
        flooded_dem = grid.read_raster(demfile[:-4] + '_pysheds_flooded_dem.tif')
    else:
        grid = Grid.from_raster(demfile[:-4]+'_clip.tif')
        dem = grid.read_raster(demfile[:-4]+'_clip.tif')
    
        # Condition DEM
        # ----------------------
        # Fill pits in DEM
        pit_filled_dem = grid.fill_pits(dem)
    
        # Fill depressions in DEM
        flooded_dem = grid.fill_depressions(pit_filled_dem)
    
        # Resolve flats in DEM
        #inflated_dem = grid.resolve_flats(flooded_dem)
    
        grid.to_raster(flooded_dem.astype('float32'), demfile[:-4] + '_pysheds_flooded_dem.tif')
    
    return flooded_dem.astype('float32')
    
def richdem_predict(demfile, keepexist):
    
    if os.path.isfile(demfile[:-4] + '_richdem_flooded_dem.tif') and keepexist:
        grid = Grid.from_raster(demfile[:-4] + '_richdem_flooded_dem.tif')
        flooded_dem = grid.read_raster(demfile[:-4] + '_richdem_flooded_dem.tif')
    else:
        grid = Grid.from_raster(demfile[:-4]+'_clip.tif')
        dem = grid.read_raster(demfile[:-4]+'_clip.tif').astype('float32')
        dem_arr = rd.rdarray(dem, no_data=-32767., geotransform=dem.affine)
        flooded_dem = rd.FillDepressions(dem_arr, in_place=False, epsilon=False)
        dem[:,:] = flooded_dem.copy().astype('float32') # restore geoinformation
        grid.to_raster(dem, demfile[:-4] + '_richdem_flooded_dem.tif')
    return flooded_dem
    
def whitebox_predict(demfile, keepexist):
    if os.path.isfile(demfile[:-4] + '_wbt_flooded_dem.tif') and keepexist:
        grid = Grid.from_raster(demfile[:-4] + '_wbt_flooded_dem.tif')
        flooded_dem = grid.read_raster(demfile[:-4] + '_wbt_flooded_dem.tif')
    else:
        wbt = WhiteboxTools()
        wbt.set_verbose_mode(False)
        if os.path.isfile(demfile[:-4] + '_wbt_flooded_dem.tif'):
            os.system('rm '+ demfile[:-4] + '_wbt_flooded_dem.tif')
        flooded_dem = wbt.fill_depressions(demfile[:-4]+'_clip.tif',  demfile[:-4] + '_wbt_flooded_dem.tif', fix_flats=False)
        while not os.path.isfile(demfile[:-4] + '_wbt_flooded_dem.tif'):
            time.sleep(1)
        grid = Grid.from_raster(demfile[:-4] + '_wbt_flooded_dem.tif')
        flooded_dem = grid.read_raster(demfile[:-4] + '_wbt_flooded_dem.tif')
        #os.system('rm '+demfile[:-4] + '_wbt_flooded_dem.tif')
    return flooded_dem
    
def plot_confusion(ax,cm):
    c=ax.imshow(cm,cmap='PuBuGn', vmin=0, vmax=1)
    fmt = '.2f'
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax, c