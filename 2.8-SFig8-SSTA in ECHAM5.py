import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.patches as patches

def bzh(x):
    return (x-np.mean(x))/np.std(x)

#%% 读取数据
P_In=np.load('./data/P_In.npy')
P_In_standardized=(P_In-np.mean(P_In))/np.std(P_In)

predicters=np.load('./data/predicters.npz')
NAD=predicters['NAD']

P_In_train=P_In_standardized[:-19]
NAD_train=NAD[:-19]

years=np.arange(1962,2019+1)

fsst=xr.open_dataset('D:/data/xibei/sst.mnmean.nc')

sst=fsst.sst.sel(time=slice('1961','2021'),lat=slice(90,-90))
sst=sst.sel(time=sst.time.dt.season=='JJA')

#%% 求逐年夏季平均和四年滑动平均
sst_JJA=sst.coarsen(time=3).mean()

sst_hd=sst_JJA.rolling(time=4).mean() #进行4年滑动平均
sst_4=sst_hd[3:].values #4年滑动平均后,前三个值是nan

#%% 计算训练时段内的SST回归场
sst_4_train=sst_4[:-19]

def regression(x,Y):
    T,I,J=Y.shape
    K=np.zeros([I,J])*np.nan
    R=np.zeros([I,J])*np.nan
    for i in range(I):
        for j in range(J):
            if ~np.isnan(Y[:,i,j]).any():
                K[i,j],_,R[i,j]=st.linregress(x,Y[:,i,j])[:3]
    return K,R

K,R=regression(P_In_train,sst_4_train)

#%%画图
fig=plt.figure(figsize=(6,5))
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 12
plt.subplots_adjust(bottom=0.15)

lines=np.arange(-0.18,0.181,0.03)
lon,lat=sst.lon,sst.lat
lonsst,latsst=np.meshgrid(lon,lat)
K_cyclic,lon_cyclic=add_cyclic_point(K, coord=lon)

cmap=plt.get_cmap('RdYlBu_r')
colors = [cmap(i) for i in (5,21,43,62,81,100,119,137,156,189,210,231,241,251)]

ax1=plt.subplot(111,projection=ccrs.PlateCarree())
ax1.set_extent([-80,10,10,75],crs=ccrs.PlateCarree())
ax1.coastlines('50m',alpha=0.8,color='0.2',zorder=2)
ax1.add_feature(cfeature.LAND.with_scale('50m'),color='0.85',zorder=1)
c1=ax1.contourf(lon_cyclic,lat,K_cyclic,lines,colors=colors,transform=ccrs.PlateCarree(),extend='both',alpha=0.4)
c2=ax1.contourf(lon_cyclic,lat,K_cyclic,lines,colors=colors,transform=ccrs.PlateCarree(),extend='both',alpha=0.85)

lon1,lon2,lat1,lat2=308,357,17,66
color='C3'
lw=2
ax1.plot([lon1,lon1],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon2,lon2],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon1,lon2],[lat1,lat1],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon1,lon2],[lat2,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())

# 创建一个矩形的路径
rect = patches.Rectangle((lon1, lat1), lon2-lon1, lat2-lat1, transform=ccrs.PlateCarree(), facecolor='none')

# 将矩形添加到图中（这样我们可以获取正确的转换）
patch = ax1.add_patch(rect)

# 使用路径来裁剪contourf图
for collection in c2.collections:
    collection.set_clip_path(patch.get_path(), transform=patch.get_transform())

ax1.set_xticks(np.arange(-80,11,15))  #指定要显示的经纬度
ax1.set_yticks(np.arange(15,76,15))
ax1.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(5))

cbar_ax=fig.add_axes([0.170,0.070,0.680,0.030])
cbar=plt.colorbar(c2,orientation='horizontal',cax=cbar_ax)

plt.savefig('./figure/Supplementary Figure 8.pdf', bbox_inches='tight')






