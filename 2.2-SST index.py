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

#%% 读取数据

P_In=np.load('./data/P_In.npy')


fsst=xr.open_dataset('D:/data/xibei/sst.mnmean.nc')

sst=fsst.sst.sel(time=slice('1961','2021'),lat=slice(90,-90))
sst=sst.sel(time=sst.time.dt.season=='JJA')

#%% 求逐年夏季平均和四年滑动平均
sst_JJA=sst.coarsen(time=3).mean()

sst_hd=sst_JJA.rolling(time=4).mean() #进行4年滑动平均
sst_4=sst_hd[3:].values #4年滑动平均后,前三个值是nan

#%% 蒙特卡洛检验
R_99=0.6182
R_95=0.4999
R_90=0.4305

#%% 计算训练时段内的SST相关系数场
P_train=P_In[:-19]
sst_4_train=sst_4[:-19]

def correlation(x,Y):
    T,I,J=Y.shape
    R=np.zeros([I,J])*np.nan
    for i in range(I):
        for j in range(J):
            if ~np.isnan(Y[:,i,j]).any():
                R[i,j]=st.pearsonr(x,Y[:,i,j])[0]
    return R

R=correlation(P_train,sst_4_train)
np.save('./data/R_train.npy',R) #训练时段内的降水内部变率与SST的相关系数场

#%% 画同期海温相关图，用于标注预测因子位置
fig=plt.figure(figsize=(13,6))
lines=[-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
lon,lat=sst.lon,sst.lat
lonsst,latsst=np.meshgrid(lon,lat)
R_cyclic,lon_cyclic=add_cyclic_point(R, coord=lon)
ax1=plt.subplot(111,projection=ccrs.PlateCarree(110))
ax1.set_extent([0,359,-60,75], crs=ccrs.PlateCarree())
ax1.coastlines('110m', alpha=0.8,color='0.2',zorder=2)
ax1.add_feature(cfeature.LAND,color='0.9',zorder=1)

cmap=palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap
color = [cmap(i) for i in (10,20,30,45,65,80,95,256-110,256-105,256-95,256-80,256-60,256-35,256-30)]
color.insert(7,(1,1,1,1)) #中间颜色改成白色

c=ax1.contourf(lon_cyclic,lat,R_cyclic,lines,
               transform=ccrs.PlateCarree(),colors=color,extend='both',alpha=0.7,zorder=2)
ax1.scatter(lonsst[np.abs(R)>R_90],latsst[np.abs(R)>R_90],marker='o', ############ R90
            color='0.1',s=2,alpha=0.4,transform=ccrs.PlateCarree(),zorder=2)


shp_CN=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_country.shp")
ax1.add_geometries(shp_CN.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=0.8,facecolor='none',zorder=2)


ax1.set_xticks(np.arange(-180,181,30))#指定要显示的经纬度
ax1.set_yticks(np.arange(-60,80,15))
ax1.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(5))
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

cbar_ax=fig.add_axes([0.91,0.21,0.01,0.58])
cbar=plt.colorbar(c,orientation='vertical',cax=cbar_ax)
cbar.set_ticks(lines)
cbar.ax.tick_params(labelsize=12)

#求预测因子序列并标出预测因子位置
def weighted_average(sst,R,R90,lon,lat,lon1,lon2,lat1,lat2):
    r=np.where(np.isnan(R),0,R) #原本陆地区域r为缺测，现权重赋值为0
    B=np.where(np.isnan(sst),0,sst) #消除掉相关系数场里的nan，不然np.average没法算
    lat_range=(lat>=lat1)&(lat<=lat2)
    lon_range=(lon>=lon1)&(lon<=lon2)

    B=B[:,lat_range,:][:,:,lon_range]
    r=r[lat_range,:][:,lon_range]
    r[np.abs(r)<R90]=0 #未通过显著性检验的格点权重为0

    T=B.shape[0]
    C=np.full(T,np.nan)
    weight=np.where(r!=0,1,0) #只有权重不为0的格点参与计算(确保参与计算的格点数为所有非缺测点数，而不是所有格点数)
    r_copy=np.where(r>0,1,-1) #不以相关系数为权重，只考虑正负

    for t in range(T):
        C[t]=np.average(B[t,:,:]*r_copy,weights=weight)

    return C


def bzh(x):
    return (x-np.mean(x))/np.std(x)

lw=2.8
C=['C3','C2','b']

###NAD
color=C[0]
lon1,lon2,lat1,lat2=308,357,17,66
NAD=bzh(weighted_average(sst_4,R,R_90,sst.lon,sst.lat,lon1,lon2,lat1,lat2)) #标准化的区域加权平均
ax1.plot([lon1,lon1],[lat1,lat2],lw=lw*1.5,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon2,lon2],[lat1,lat2],lw=lw*1.5,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon1,lon2],[lat1,lat1],lw=lw*1.5,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon1,lon2],[lat2,lat2],lw=lw*1.5,c=color,transform=ccrs.PlateCarree())
ax1.text((lon1+lon2)/2,(lat1+lat2)/2,'NAD',c=color,fontsize='xx-large',transform=ccrs.PlateCarree(),ha='center',va='center')
print('NAD与P_In相关系数：{:.4f},{:.4f}'.format(st.pearsonr(NAD[:-19],P_In[:-19])[0],st.pearsonr(NAD,P_In)[0]))

###PDO
color=C[1]
lon1,lon2,lat1,lat2=123,283,-33,60
PDO=bzh(weighted_average(sst_4,R,R_90,sst.lon,sst.lat,lon1,lon2,lat1,lat2)) #标准化的区域加权平均
ax1.plot([lon1,lon1],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon2,lon2],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon1,lon2],[lat1,lat1],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon1,lon2],[lat2,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.text((lon1+lon2)/2,(lat1+lat2)/2,'PDO',c=color,fontsize='xx-large',transform=ccrs.PlateCarree(),ha='center',va='center')
print('PDO与P_In相关系数：{:.4f},{:.4f}'.format(st.pearsonr(PDO[:-19],P_In[:-19])[0],st.pearsonr(PDO,P_In)[0]))

###IO
color=C[2]
lon1,lon2,lat1,lat2=34,120,-41,25
IO=bzh(weighted_average(sst_4,R,R_90,sst.lon,sst.lat,lon1,lon2,lat1,lat2)) #标准化的区域加权平均
ax1.plot([lon1,lon1],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon2,lon2],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon1,lon2],[lat1,lat1],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.plot([lon1,lon2],[lat2,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
ax1.text((lon1+lon2)/2,(lat1+lat2)/2,'IO',c=color,fontsize='xx-large',transform=ccrs.PlateCarree(),ha='center',va='center')
print('IO与P_In相关系数：{:.4f},{:.4f}'.format(st.pearsonr(IO[:-19],P_In[:-19])[0],st.pearsonr(IO,P_In)[0]))


np.savez('./data/predicters.npz',IO=IO,PDO=PDO,NAD=NAD) #预测因子序列

#%% 画预测因子序列图
Ps=['NWCP-I','NAD','PDO','IO']
P=[P_In,NAD,PDO,IO]
year=np.arange(1962,2020)
plt.figure(figsize=(10,4))
ax=plt.subplot(111)
for i,x in enumerate(P):
    if i==0:
        plt.plot(year,bzh(x),lw=3,label=Ps[i],c='k',alpha=0.8)
    else:
        plt.plot(year,x,lw=2,label=Ps[i],c=C[i-1],alpha=0.5)
# plt.grid()
plt.axvline(2002.5,color='y') #年代际序列训练截止到2000年，对应训练时段到2002年
# plt.axvspan(2002.5,2021,color='')
plt.xlim(1960,2020)
plt.xticks(np.arange(1960,2021,5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
plt.legend()
plt.savefig('./figure/3_1-预测因子序列图', bbox_inches='tight',dpi=300)

#%% 计算每个预测因子之间的相关系数

print('{:>5s}\t{:>5s}\t{:>5s}\t{:>5s}\t{:>5s}'.format('','NWCP-I','NAD','PDO','IO'))

for i,x in enumerate(P):
    print('{:>5s}\t'.format(Ps[i]),end='')
    for y in P:
        r=st.pearsonr(x[:-19],y[:-19])[0]
        if R_95>r:
            print('{:.2f}\t'.format(r),end=' ')
        elif R_95<r<R_99:
            print('{:.2f}*\t'.format(r),end=' ')
        else:
            print('{:.2f}**\t'.format(r),end=' ')
    print()




