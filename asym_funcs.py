import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import os
xr.set_options(keep_attrs=True)
from scipy.fft import fft, ifft
from scipy.interpolate import splev, splrep, sproot


a_earth = 510072000./1e6 # million km^2

firstofmonthind = [1,32,60,91,121,152,182,213,244,274,305,335]
firstofmonthlabel = ['01-Jan','01-Feb','01-Mar','01-Apr','01-May','01-Jun','01-Jul',
                    '01-Aug','01-Sep','01-Oct','01-Nov','01-Dec']

shfirstofmonthind = [1,32,63,93,124,154,185,216,244,275,305,335]
shfirstofmonthlabel = ['01-Jul','01-Aug','01-Sep','01-Oct','01-Nov','01-Dec','01-Jan','01-Feb','01-Mar','01-Apr','01-May','01-Jun']


def get_annual_harmonic (xdata, ydata):
    """
    Compute the annual harmonic using FFT
    """
    
    N = len(ydata)
    yinv = fft(ydata)/N
    
    annual_harmonic = yinv[0] + np.real(yinv[N-1] + yinv[1])*np.cos(2.*np.pi*xdata) + np.imag(yinv[N-1] - yinv[1])*np.sin(2.*np.pi*xdata)
    
    return annual_harmonic

def xr_get_annual_harmonic (xdata, ydata, dim):
    
    annual_harmonic = xr.apply_ufunc(get_annual_harmonic,
                       xdata, ydata,
                       input_core_dims  = [[dim], [dim]], 
                       output_core_dims = [[dim]],
                       vectorize=True)    


    return annual_harmonic


def fourier_decomp (xdata, ydata, n):
    
    N = len(ydata)
    yinv = fft(ydata)/N

    if (N % 2) ==0:
        N2, N3 = int(N/2), int(N/2)
    else:
        N2, N3 = int((N-1)/2), int((N+1)/2)
        
    ak0 = 2.*yinv[0]

    ak = np.real(yinv[N2+1:][::-1] + yinv[1:N3] )
    bk = np.imag(yinv[N2+1:][::-1] - yinv[1:N3]  )

    output = ak0/2.

    for i in np.arange(1,n+1,1):
        output = output + ak[i-1]*np.cos(2.*np.pi*i*xdata) + bk[i-1]*np.sin(2.*np.pi*i*xdata)
    
    return np.real(output)

def xr_fourier_decomp(first_samples, second_samples, n, dim):
    """
    Apply the fourier_decomp function to an xarray Dataset
    """
    y2 = xr.apply_ufunc(fourier_decomp,
                       first_samples, second_samples, n,
                       input_core_dims  = [[dim],[dim],[]], 
                       output_core_dims = [[dim]],
                       vectorize=True)    
    return y2
                                                                                                                          
def _nanlinregress(x, y):
    """
    Calls scipy linregress only on finite numbers of x and y
    """
    
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        # empty arrays passed to linreg raise ValueError:
        # force returning an object with nans:
        return scipy.stats.linregress([np.nan], [np.nan])
    return scipy.stats.linregress(x[finite], y[finite])



def linregress(first_samples, second_samples, dim):
    """
    Apply the _nanlinregress function to an xarray Dataset
    """
    slope, intercept, r_value, p_value, std_err = xr.apply_ufunc(_nanlinregress,
                       first_samples, second_samples,
                       input_core_dims  = [[dim], [dim]], 
                       output_core_dims = [[],[],[],[],[]],
                       vectorize=True)    
    return slope, intercept, r_value, p_value, std_err






def asym_1d (mydata, tolerance=0.):
    """
    Compute the difference between the length of the growth and retreat seasons
    Seasons are defined time of maximum and minimum occurrence
    
    Note that this only works for extrema that are within the time axis - 
     i.e. not if you have an extrema near the end and beginning of the period
     
    The tolerance is expressed in terms of a fraction of the amplitude
    """
    
    nt = len(mydata)
    
    if tolerance == 0.:
        dmin =  np.argwhere(mydata<=np.nanmin(mydata)).mean()
        dmax =  np.argwhere(mydata>=np.nanmax(mydata)).mean()
        
    
    else: # more general case
        amp = np.nanmax(mydata)-np.nanmin(mydata)
        if amp>0:
            tolerance = tolerance*amp


            mymins = np.argwhere(mydata<=(np.nanmin(mydata)+tolerance))
            if len(np.shape(mymins))==2:
                mymins = mymins[:,0]
            if len(mymins)>1:
                if max(abs(np.diff(mymins)))>nt/3:
                    mymins = sorted(np.where(mymins<nt/2,mymins+nt,mymins))
            dmin = np.mean(np.arange(min(mymins),max(mymins)+1,1))
            if dmin>nt:
                dmin = dmin - nt

            mymaxs = np.argwhere(mydata>=(np.nanmax(mydata)-tolerance))[:,0]
            if len(mymaxs)>1:
                if max(abs(np.diff(mymaxs)))>nt/3:
                    mymaxs = sorted(np.where(mymaxs<nt/2,mymaxs+nt,mymaxs))
            dmax = np.mean(np.arange(min(mymaxs),max(mymaxs)+1,1))
            if dmax>nt:
                dmax = dmax - nt
        else:
            dmax, dmin = np.nan, np.nan



    if dmax>dmin:
        growth = dmax - dmin
        retreat = nt - growth
    else:
        retreat = dmin - dmax
        growth = nt - retreat

    asym = growth - retreat
   
    return asym, dmin, dmax


def asym_xr (ds, dim, tolerance=0.) :                                                                                                                     
    """
    Apply the asym_1d function to an xarray Dataset
    """ 
    asym, dmin, dmax = xr.apply_ufunc(asym_1d,
                   ds, tolerance,
                   input_core_dims  = [[dim],[]], 
                   output_core_dims = [[],[],[]],
                   vectorize=True)

    
    return asym, dmin, dmax
                                                                                                                          
                                                                                                                          


def grid_area_regll(lat,lon):
    """
    Compute the area of grid cells on a regular lat lon grid, given lats and lons
    """ 
    to_rad = 2. *np.pi/360.
    r_earth = 6371.22 # km
    con = r_earth*to_rad
    clat = np.cos(lat*to_rad)
    dlon = lon[2] - lon[1]
    dlat = lat[2] - lat[1]
    dx = con*dlon*clat
    dy = con*dlat
    dxdy = dy*dx
    garea = np.swapaxes(np.tile(dxdy,(len(lon),1)),0,1)
    latl = np.swapaxes(np.tile(lat,(len(lon),1)),0,1)
    nh_area = np.where(latl<0.,0.,garea)
    sh_area = np.where(latl>0.,0.,garea)
    
    return garea, nh_area, sh_area


def simple_model (nspace=800, nts=1000, mysolar=0., mycw=9.8, myA=138.6, dur=100, myFb=5., myB=2.1, forc_xt = 0.):
    """
    Run the simple model globally. Can vary parameters, resolution and run length as input to this function
    """                                                                                                                                                                                                                                  
    # Physical parameters
    S1 = 338.;     # insolation seasonal dependence (W m^-2)
    A  = myA #193  # OLR when T = 0 (W m^-2)
    B  = myB #2.1  # OLR temperature dependence (W m^-2 K^-1)
    cw = mycw      # ocean mixed layer heat capacity (W yr m^-2 K^-1)
    S0 = 420.      # insolation at equator  (W m^-2)
    S2 = 240.      # insolation spatial dependence (W m^-2)
    a0 = 0.7       # ice-free co-albedo at equator
    a2 = 0.1       # ice=free co-albedo spatial dependence
    Fb = myFb #4   # heat flux from ocean below (W m^-2)
  
    # Time stepping parameters
    #dur          # of years for the whole run
    n  = nspace;  # of evenly spaced latitudinal gridboxes (pole to pole)
    nt = nts;     # of timesteps per year (approx lower limit of stability) 
    dt = 1/nt;
    ty = np.arange(dt/2,1+dt/2,dt)
    
    print("Arguments: nx = {0}, nt = {1}, duration = {2} years".format(nspace,nts,dur))
    print("Parameters: c_w = {}, A = {}, Fb = {}, B = {}".format(cw, A, Fb, B))
    print("Forcing: solar {}, additional {}".format(np.max(np.abs(mysolar==0.))>0, np.max(np.abs(forc_xt))>0))
    
    
    #Spatial Grid -------------------------------------------------------------
    dx = 2.0/n    #grid box width
    x = np.arange(-1+dx/2,1+dx/2,dx) #native grid
    lat = np.arcsin(x)*180./np.pi
  
    ##Seasonal forcing (WE15 eq.3)
    if np.all(mysolar==0.): # if an insolation field is not provided, use the idealized function
        S = (np.tile(S0-S2*x**2,[nt,1])- np.tile(S1*np.cos(2*np.pi*ty),[n,1]).T*np.tile(x,[nt,1])); 
    else:
        S = mysolar.T
    S = np.vstack((S,S[0,:]))
    
    Amod = np.zeros([nt,n])
    if np.all(forc_xt == 0.):
        Amod[:,:] = A
    else:
        Amod[:,:] = - forc_xt[:,:].T + A
    Amod = np.vstack((Amod,Amod[0,:]))
      
    alpha = a0-a2*x**2      # open water albedo

    #Set up output arrays, saving all timesteps of all years
    Tfin  = np.zeros([dur,nt,n])
   
    #Initial conditions ------------------------------------------------------
    T = 7.5+20*(1-2*x**2);
    
    #Loop over Years ---------------------------------------------------------
    for years in range(0,dur):
        #Loop within One Year-------------------------------------------------
        for i in range(0,int(nt)):
            
            #forcing
            T = T + (dt/cw)*(alpha*S[i+1] - (Amod[i+1]+B*T)+Fb)                
            Tfin[years,i,:] = T
                   
    T_all = xr.DataArray(Tfin,dims=('year','time','lat'),coords = {'year':np.arange(1,dur+1,1), 'time':ty, 'lat':lat}).to_dataset(name='T')
    
    return T_all





def fix_lp_yr(ds):
    """
    If there are 366 days, interpolate them over 365 days

    """
    if len(ds.time) == 366:
        
        newtime = [f for f in ds.time.values if '02-29' not in str(f)]
        ds['time'] = np.linspace(1,365,366)
        ds = ds.interp(time=np.linspace(1,365,365))
        ds['time'] = newtime
              
    return ds



def xr_reshape(A, dim, newdims, coords):
    """ Reshape DataArray A to convert its dimension dim into sub-dimensions given by
    newdims and the corresponding coords.
    Example: Ar = xr_reshape(A, 'time', ['year', 'month'], [(2017, 2018), np.arange(12)]) """

    # Create a pandas MultiIndex from these labels
    ind = pd.MultiIndex.from_product(coords, names=newdims)

    # Replace the time index in the DataArray by this new index,
    A1 = A.copy()

    A1.coords[dim] = ind

    # Convert multiindex to individual dims using DataArray.unstack().
    # This changes dimension order! The new dimensions are at the end.
    A1 = A1.unstack(dim)

    # Permute to restore dimensions
    i = A.dims.index(dim)
    dims = list(A1.dims)

    for d in newdims[::-1]:
        dims.insert(i, d)

    for d in newdims:
        _ = dims.pop(-1)


    return A1.transpose(*dims)


def cw_to_mld(cw):

    c=4000.
    p=1025.
    dt = 31536000.
    return float(cw)*dt/(c*p)

def mld_to_cw(mld):

    c=4000.
    p=1025.
    dt = 31536000.
    return float(mld)*c*p/dt
