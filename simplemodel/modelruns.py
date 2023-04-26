import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, './../')
import asym_funcs as af
import os

mydir = './output/ODEn800_'

myfiles = sorted([f for f in os.listdir('./forcingdata') if ('forcing' in f and '.nc' in f and ('insol' in f or 'SW' in f) and 'zonal' not in f)])
print(myfiles)
for myfile in myfiles:
    print(myfile)
    sol = xr.open_dataset('./forcingdata/'+myfile).S.values
    print(sol.shape)
    ds = af.simple_model(mysolar=sol)
    myname = myfile.split('_n8')[0][8:]
    print(myname) 
    ds['names'] = myname
    ds = ds.set_coords('names')
    ds = ds.isel(year=slice(-20,None)).mean(dim='year')
    ds.to_netcdf(mydir+myname+'.nc')

sol = xr.open_dataset('forcingdata/forcing_TOAinsolation_n800.nc').S.values
hc = xr.open_dataset('forcingdata/forcing_ERA5_HC_n800.nc').HC.values
myname = 'TOAinsolationPLUSheatconv'
print(myname) 
print(sol.shape, hc.shape)
ds = af.simple_model(mysolar=sol, forc_xt = hc)
ds['names'] = myname
ds = ds.set_coords('names')
ds = ds.isel(year=slice(-20,None)).mean(dim='year')
ds.to_netcdf(mydir+myname+'.nc')

sol = xr.open_dataset('forcingdata/forcing_TOAinsolation_n800.nc').S.values
hc = xr.open_dataset('forcingdata/forcing_CERES_SFC-LW-down_anom_n800.nc').L.values
myname = 'TOAinsolationPLUSanomLWCERES'
print(myname)
print(sol.shape, hc.shape)
ds = af.simple_model(mysolar=sol, forc_xt = hc)
ds['names'] = myname
ds = ds.set_coords('names')
ds = ds.isel(year=slice(-20,None)).mean(dim='year')
ds.to_netcdf(mydir+myname+'.nc')

for mld in [10,50,100,200]:
    cw = af.mld_to_cw(mld)
    myname = 'TOAinsolation_MLD'+str(mld)+'m'
    ds = af.simple_model(mysolar=sol, mycw = cw)
    ds['cw'] = cw
    ds['names'] = myname
    ds = ds.set_coords('names')
    ds = ds.set_coords('cw')
    ds = ds.isel(year=slice(-20,None)).mean(dim='year')
    ds.to_netcdf(mydir+myname+'.nc')
    
    
## Zonal variations in SW forcing
#sol = xr.open_dataset('forcingdata/forcing_CERES_SFC-SW-down_zonal-and-meridional_n800.nc')
#mydir = './output/zonal/ODEn800_' 
#for lon in sol.lon.values:
#    myname = 'CERES_SFC-SW-down_lon'+str(lon)
#    ds = af.simple_model(mysolar=sol.S.sel(lon=lon).squeeze())
#    ds['lon'] = lon
#    ds['names'] = myname
#    ds = ds.set_coords('names')
#    ds = ds.set_coords('lon')
#    ds = ds.isel(year=slice(-20,None)).mean(dim='year')
#    ds.to_netcdf(mydir+myname+'.nc')




