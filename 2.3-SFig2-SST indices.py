import numpy as np
import xarray as xr
from metpy.units import units
import scipy.signal as scisig
import matplotlib.pyplot as plt
import scipy.stats as st
import metpy.calc as mpcalc
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from matplotlib import ticker
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature
import matplotlib.colors as colors
import palettable
from cartopy.util import add_cyclic_point
from toolbox import gaitu

def sliding_correlation(x,y,window_size,alpha):
    """
    计算两个时间序列的滑动相关系数。
    x, y: numpy数组，代表两个时间序列。
    window_size: 滑动窗口的大小。
    """
    if len(x)!=len(y):
        raise ValueError("两个时间序列长度必须相同")

    n=len(x)
    correlations=[]
    for i in range(n-window_size+1):
        x_window=x[i:i+window_size]
        y_window=y[i:i+window_size]
        correlation=st.pearsonr(x_window,y_window)[0]
        correlations.append(correlation)

    return correlations

def bzh(x):
    return (x-np.mean(x))/np.std(x)

#%% 读取数据
fsst=xr.open_dataset('D:/data/xibei/sst.mnmean.nc')
sst=fsst.sst.sel(time=slice('1961','2021'),lat=slice(90,-90)) #用来读取经纬度

R_train=np.load('./data/R_train.npy') #训练时段内的降水内部变率与SST的相关系数场

P_In=np.load('./data/P_In.npy')
P_In_standardized=(P_In-np.mean(P_In))/np.std(P_In)

predicters=np.load('./data/predicters.npz')
NAD,PDO,IO=predicters['NAD'],predicters['PDO'],predicters['IO']

years=np.arange(1962,2019+1)

#%%计算相关系数
r_NAD=st.pearsonr(P_In,NAD)[0]
r_NAD_train=st.pearsonr(P_In[years<=2000],NAD[years<=2000])[0]

r_PDO=st.pearsonr(P_In,PDO)[0]
r_PDO_train=st.pearsonr(P_In[years<=2000],PDO[years<=2000])[0]

r_IO=st.pearsonr(P_In,IO)[0]
r_IO_train=st.pearsonr(P_In[years<=2000],IO[years<=2000])[0]

#%%画图
plt.figure(figsize=(10,4))
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 12

Ps=['NWCP-I','NAD','PDO','IO']
P=[P_In,NAD,PDO,IO]
C=['C3','C2','b']

ax=plt.subplot(111)
for i,x in enumerate(P):
    if i==0:
        plt.plot(years,bzh(x),lw=3,label=Ps[i],c='royalblue',alpha=0.7)
    else:
        plt.plot(years,x,lw=2,label=Ps[i],c=C[i-1],alpha=0.9)

plt.ylim(-4,3)
plt.ylabel('Intensity (standardized)')
plt.xlim(1961,2021)
plt.xlabel('Year')
ax1=plt.gca()
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
plt.grid(ls='--')
plt.axhline(0,ls='--',c='grey')
ax1.axvspan(2002.5,2021,color='lightblue',alpha=0.3)
ax1.axvspan(1960,2002.5,color='#FFDE91',alpha=0.2)
plt.legend(loc='upper left')

plt.text(1985,-1.8,'1961-2002',fontsize=10,ha='center',color='k')
plt.text(1993,-1.8,'1961-2021',fontsize=10,ha='center',color='k')

plt.text(1970,-2.3,'Corr(NWCP-I, NAD)',fontsize=10,ha='left',color='C3')
plt.text(1985,-2.3,f'{r_NAD_train:.2f}',fontsize=10,ha='center',color='C3')
plt.text(1993,-2.3,f'{r_NAD:.2f}',fontsize=10,ha='center',color='C3')

plt.text(1970,-2.8,'Corr(NWCP-I, PDO)',fontsize=10,ha='left',color='C2')
plt.text(1985,-2.8,f'{r_PDO_train:.2f}',fontsize=10,ha='center',color='C2')
plt.text(1993,-2.8,f'{r_PDO:.2f}',fontsize=10,ha='center',color='C2')

plt.text(1970,-3.3,'Corr(NWCP-I, IO)',fontsize=10,ha='left',color='b')
plt.text(1985,-3.3,f'{r_IO_train:.2f}',fontsize=10,ha='center',color='b')
plt.text(1993,-3.3,f'{r_IO:.2f}',fontsize=10,ha='center',color='b')

plt.savefig('./figure/Supplementary Figure 2.pdf', bbox_inches='tight')








