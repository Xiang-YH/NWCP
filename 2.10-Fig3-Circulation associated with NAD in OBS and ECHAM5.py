import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from cartopy.io.shapereader import Reader
from cartopy.util import add_cyclic_point
import pandas as pd

def regression(x,Y):
    T,I,J=Y.shape
    K=np.zeros([I,J])*np.nan
    R=np.zeros([I,J])*np.nan
    for i in range(I):
        for j in range(J):
            if ~np.isnan(Y[:,i,j]).any():
                K[i,j],_,R[i,j]=st.linregress(x,Y[:,i,j])[:3]
    return K,R

#%%读取数据
P_In=np.load('./data/P_In.npy')

predicters=np.load('./data/predicters.npz')
NAD=predicters['NAD']

fuvz=xr.open_dataset('D:/data/xibei/ERA5-UVZ-1961_2021-25_25.nc')

u200=fuvz.u.sel(time=slice('1961','2021'),latitude=slice(90,0),level=200)
u200=u200.sel(time=u200.time.dt.season=='JJA')
v200=fuvz.v.sel(time=slice('1961','2021'),latitude=slice(90,0),level=200)
v200=v200.sel(time=v200.time.dt.season=='JJA')
z200=fuvz.z.sel(time=slice('1961','2021'),latitude=slice(90,0),level=200)/9.8
z200=z200.sel(time=z200.time.dt.season=='JJA')

years=np.arange(1962,2020)
lon,lat=z200.longitude,z200.latitude

#%% 求逐年夏季平均和风场平均态
z200_JJA=z200.coarsen(time=3).mean()

u200_mean=u200.values.mean(0)
v200_mean=v200.values.mean(0)

#%% 四年滑动平均
z200_hd=z200_JJA.rolling(time=4).mean()  #进行4年滑动平均
z200_4=z200_hd[3:].values  #4年滑动平均后,前三个值是nan

#%%提取训练时段
#训练时段1961-2002年，对应年代际序列的范围时1962-2000年
years_train=years[:-19]
print(years_train)

P_train=P_In[:-19]

z200_4_train=z200_4[:-19]

NAD_train=NAD[:-19]

#%% 蒙特卡洛检验
R_95=0.4999
R_90=0.4305

#%% 计算预测因子同期回归环流
def TN(za,u_mean,v_mean,lon,lat,p0):
    """
    计算波活动通量的x和y分量。

    :param za: 参数“za”代表位势高度异常。它是对给定位置和时间的位势高度与其平均值的偏差的度量。
    :param u_mean: 平均纬向风速，单位为米/秒。
    :param v_mean: 平均经向风速，单位为米/秒。
    :param lon: 网格点的经度值（以度为单位）。
    :param lat: 网格点的纬度值（以度为单位）。
    :param p0: p0 是表面压力，单位为 hPa（百帕斯卡）。
    :return: 使用给定参数计算的波活动通量的x和y分量。
    """
    a = 6400000 #地球半径
    omega = 7.292e-5 #自转角速度
    dlon = (np.gradient(lon)*np.pi/180.0).reshape((1,-1))
    dlat = (np.gradient(lat)*np.pi/180.0).reshape((-1,1))
    f = np.array(2*omega*np.sin(lat*np.pi/180.0)).reshape((-1,1)) #Coriolis parameter: f=2*omgega*sin(lat )
    cos_lat = (np.cos(np.array(lat)*np.pi/180)).reshape((-1,1)) #cos(lat)
    u_c=u_mean
    v_c=v_mean
    g=9.8
    psi_p = g*za/f #Pertubation stream-function
    #5 partial differential terms
    dpsi_dlon = np.gradient(psi_p, axis=1)/dlon
    dpsi_dlat = np.gradient(psi_p, axis=0)/dlat
    d2psi_dlon2=np.gradient(dpsi_dlon, axis=1)/dlon
    d2psi_dlat2=np.gradient(dpsi_dlat, axis=0)/dlat
    d2psi_dlondlat = np.gradient(dpsi_dlon, axis=0)/dlat
    termxu=dpsi_dlon*dpsi_dlon-psi_p*d2psi_dlon2
    termxv=dpsi_dlon*dpsi_dlat-psi_p*d2psi_dlondlat
    termyv=dpsi_dlat*dpsi_dlat-psi_p*d2psi_dlat2
    #coefficient
    p= p0/1000
    magU=np.sqrt(u_c**2+v_c**2)
    coeff=p*cos_lat/(2*a*a*magU)
    #x-component of TN-WAF
    px = coeff * ((u_c/cos_lat/cos_lat)*termxu+v_c*termxv/cos_lat)
    #y- component of TN-WAF
    py = coeff * ((u_c/cos_lat)*termxv+v_c*termyv)
    return px,py


k_z200,r_z200=regression(NAD_train,z200_4_train)
TNx_obs,TNy_obs=TN(k_z200,u200_mean,v200_mean,lon,lat,200)
TNx_obs[22:,:]=np.nan
TNy_obs[22:,:]=np.nan

#%%观测里的风暴轴和dzdt
f_st=xr.open_dataset('./data/NAD_storm_track.nc')
storm_track_mean_state=f_st['storm_track_mean_state']
k_storm_track=f_st['k_storm_track']
r_storm_track=f_st['r_storm_track']
lon_st,lat_st=f_st.lon,f_st.lat

f_dzdt=xr.open_dataset('./data/NAD_dzdt.nc')
k_dzdt=f_dzdt['k_dzdt']
r_dzdt=f_dzdt['r_dzdt']
lon_dzdt,lat_dzdt=f_dzdt.lon,f_dzdt.lat


#%%模式Hgt200响应
f_Ctrl=xr.open_dataset('D:/data/xibei/ECHAM/ECHAM5/Ctrl/XB100ctrl_198001-199912.mon.nc')
f_Sen=xr.open_dataset('D:/data/xibei/ECHAM/ECHAM5/Sen/XB100sen_198001-199912.mon.nc')

lon_mod_z,lat_mod_z=f_Ctrl.lon,f_Ctrl.lat

def calc_diff(varname,**kwargs):
    Var_Ctrl=f_Ctrl[varname].sel(**kwargs)
    Var_Sen=f_Sen[varname].sel(**kwargs)
    Var_diff=Var_Sen-Var_Ctrl
    return Var_diff

z200_diff=calc_diff('geopoth',lev=20000)
u200_diff=calc_diff('u',lev=20000)
v200_diff=calc_diff('v',lev=20000)

u200_ctrl=f_Ctrl['u'].sel(lev=20000)
u200_ctrl_JJA=u200_ctrl.sel(time=u200_ctrl.time.dt.season=='JJA')
u200_ctrl_mean=u200_ctrl_JJA.mean('time').values

v200_ctrl=f_Ctrl['v'].sel(lev=20000)
v200_ctrl_JJA=v200_ctrl.sel(time=v200_ctrl.time.dt.season=='JJA')
v200_ctrl_mean=v200_ctrl_JJA.mean('time').values

def JJA_mean(Var_diff):
    new_time = pd.date_range(start='2001-01-01', end='2020-12-01', freq='MS')
    Var_diff['time'] = new_time
    Var_diff=Var_diff.sel(time=slice('2006','2020'))
    Var_diff_season=Var_diff.groupby('time.season').mean()
    Var_diff_JJA=Var_diff_season.sel(season='JJA')
    return Var_diff_JJA.values

z200_diff_JJA=JJA_mean(z200_diff)
u200_diff_JJA=JJA_mean(u200_diff)
v200_diff_JJA=JJA_mean(v200_diff)

TNx_mod,TNy_mod=TN(z200_diff_JJA,u200_ctrl_mean,v200_ctrl_mean,lon_mod_z,lat_mod_z,200)
TNx_mod[22:,:]=np.nan
TNy_mod[22:,:]=np.nan

#%%模式海温强迫
f_mod_ssta=xr.open_dataset('D:/data/xibei/ECHAM/ECHAM5/test_forcing_ssta.nc')
ssta=f_mod_ssta.ssta[6]

#%%模式瞬变涡旋响应
f_mod_st=xr.open_dataset('./data/ECHAM_storm_track_diff.nc')
storm_track_diff=f_mod_st['storm_track_diff']
lon_mod_st,lat_mod_st=f_mod_st.lon,f_mod_st.lat

#%%画图
fig=plt.figure(figsize=(12,9.5))
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 15

#---------------------------------------------(a)------------------------------------------
k_z200_cyclic,lon_cyclic=add_cyclic_point(k_z200, coord=lon)
r_z200_cyclic,_=add_cyclic_point(r_z200, coord=lon)
Lon_cyclic,Lat=np.meshgrid(lon_cyclic,lat)
lines=np.arange(-12,12.1,1.2)

ax1=fig.add_axes([0.099,0.726,0.773,0.234],projection=ccrs.PlateCarree(0))
ax1.set_extent([-70,180,20,80], crs=ccrs.PlateCarree())
ax1.set_title('a',loc="left",fontweight='bold')
ax1.coastlines('110m', alpha=0.8,color='0.2')
#
cmap=plt.cm.get_cmap('PuOr_r')
color=[cmap(i) for i in (5,10,20,30,40,50,60,70,95,115,256-120,256-110,256-90,256-80,256-65,256-58,256-52,256-45,256-38,256-30,256-15)]
c=ax1.contourf(lon_cyclic,lat,k_z200_cyclic,lines,
               transform=ccrs.PlateCarree(),colors=color,extend='both',alpha=0.8)
TNx_obs[TNx_obs**2+TNy_obs**2<0.0005]=np.nan
quiv1=ax1.quiver(lon[::2],lat[::2],TNx_obs[::2,::2],TNy_obs[::2,::2],transform=ccrs.PlateCarree()
             ,headwidth=4,headlength=6,scale=4,width=0.0021,color='darkgreen',alpha=0.95)
ax1.quiverkey(quiv1,0.95,1.05,0.1,"0.1",fontproperties={'size':11},labelpos='W')
ax1.scatter(Lon_cyclic[np.abs(r_z200_cyclic)>R_90],Lat[np.abs(r_z200_cyclic)>R_90],marker='o', ############ R90
            color='w',s=2,alpha=0.4,transform=ccrs.PlateCarree())

shp_CN=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_country.shp")
ax1.add_geometries(shp_CN.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=0.8,facecolor='none')
shp_NWC=Reader("./data/NWC.shp")
ax1.add_geometries(shp_NWC.geometries(),crs=ccrs.PlateCarree(),edgecolor='darkred',linewidths=1.3,facecolor='none')

ax1.set_xticks(np.arange(-70,181,20))  #指定要显示的经纬度
ax1.set_yticks(np.arange(20,81,10))
ax1.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(5))


cbar_ax=fig.add_axes([0.170,0.684,0.600,0.010])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)
cbar.set_ticks(lines)

#---------------------------------------------(b)------------------------------------------
lines=np.array([-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8])
ax2=fig.add_axes([0.099,0.409,0.238,0.229],projection=ccrs.PlateCarree(0))
ax2.set_title('b',loc="left",fontweight='bold')
ax2.set_extent([-70,20,10,75],crs=ccrs.PlateCarree())
ax2.coastlines('50m',alpha=0.8,color='0.2')
ax2.contour(lon_st,lat_st,storm_track_mean_state,np.arange(16,31,2),colors='k',transform=ccrs.PlateCarree(),alpha=0.5,linewidths=1)
c=ax2.contourf(lon_st,lat_st,k_storm_track,lines,cmap='RdBu_r',transform=ccrs.PlateCarree(),extend='both',alpha=0.8)
Lon_st,Lat_st=np.meshgrid(lon_st,lat_st)
ax2.scatter(Lon_st[np.abs(r_storm_track)>R_90],Lat_st[np.abs(r_storm_track)>R_90],marker='o', ############ R90
            color='w',s=2,alpha=0.4,transform=ccrs.PlateCarree())
shp_CN=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_country.shp")
ax2.add_geometries(shp_CN.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=0.8,facecolor='none')
ax2.set_xticks(np.arange(-70,21,15))  #指定要显示的经纬度
ax2.set_yticks(np.arange(15,76,15))
ax2.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax2.yaxis.set_major_formatter(LatitudeFormatter())
ax2.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(5))

cbar_ax=fig.add_axes([0.102,0.373,0.230,0.007])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)

#---------------------------------------------(c)------------------------------------------
ax3=fig.add_axes([0.366,0.409,0.238,0.229],projection=ccrs.PlateCarree(0))
ax3.set_extent([-70,20,10,75],crs=ccrs.PlateCarree())
lines=np.array([-2.7,-2.4,-2.1,-1.8,-1.5,-1.2,-0.9,-0.6,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7])
ax3.set_title('c',loc="left",fontweight='bold')
ax3.coastlines('50m',alpha=0.8,color='0.2')
c=ax3.contourf(lon_dzdt,lat_dzdt,k_dzdt,lines,
               transform=ccrs.PlateCarree(),extend='both',alpha=0.8,cmap='PuOr_r')
Lon_dzdt,Lat_dzdt=np.meshgrid(lon_dzdt,lat_dzdt)
ax3.scatter(Lon_dzdt[np.abs(r_dzdt)>R_90],Lat_dzdt[np.abs(r_dzdt)>R_90],marker='o', ############ R90
            color='w',s=2,alpha=0.4,transform=ccrs.PlateCarree())
shp_CN=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_country.shp")
ax3.add_geometries(shp_CN.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=0.8,facecolor='none')
ax3.set_xticks(np.arange(-70,21,15))  #指定要显示的经纬度
ax3.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax3.yaxis.set_major_formatter(LatitudeFormatter())
ax3.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax3.yaxis.set_minor_locator(mticker.MultipleLocator(5))

cbar_ax=fig.add_axes([0.369,0.373,0.230,0.007])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)

#---------------------------------------------(d)------------------------------------------
ax4=fig.add_axes([0.633,0.409,0.238,0.229],projection=ccrs.PlateCarree(0))
ax4.set_extent([-70,20,10,75],crs=ccrs.PlateCarree())
ax4.set_title('d',loc="left",fontweight='bold')
ax4.coastlines('50m',alpha=0.8,color='0.2')

lines=np.array([-4.8,-4.2,-3.6,-3.0,-2.4,-1.8,-1.2,-0.6,0.6,1.2,1.8,2.4,3.0,3.6,4.2,4.8])
c=ax4.contourf(lon_mod_st,lat_mod_st,storm_track_diff,lines,cmap='RdBu_r',transform=ccrs.PlateCarree(),extend='both',alpha=0.8)

ax4.set_xticks(np.arange(-70,21,15))  #指定要显示的经纬度
ax4.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax4.yaxis.set_major_formatter(LatitudeFormatter())
ax4.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax4.yaxis.set_minor_locator(mticker.MultipleLocator(5))

cbar_ax=fig.add_axes([0.636,0.373,0.230,0.007])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)



#---------------------------------------------(e)------------------------------------------
z200_diff_cyclic,lon_mod_z_cyclic=add_cyclic_point(z200_diff_JJA, coord=lon_mod_z)
lines=np.arange(-30,30.1,3)

ax5=fig.add_axes([0.099,0.079,0.773,0.234],projection=ccrs.PlateCarree(0))
ax5.set_extent([-70,180,20,80], crs=ccrs.PlateCarree())
ax5.set_title('e',loc="left",fontweight='bold')
ax5.coastlines('110m', alpha=0.8,color='0.2')

cmap=plt.cm.get_cmap('PuOr_r')
color=[cmap(i) for i in (10,18,26,34,42,54,64,78,92,104,116,128,139,151,162,174,186,197,209,221,232,244,256)]
c=ax5.contourf(lon_mod_z_cyclic,lat_mod_z,z200_diff_cyclic,lines,
               transform=ccrs.PlateCarree(),cmap=cmap,extend='both',alpha=0.8)
TNx_mod[TNx_mod**2+TNy_mod**2<0.01]=np.nan
quiv1=ax5.quiver(lon_mod_z[::2],lat_mod_z[::2],TNx_mod[::2,::2],TNy_mod[::2,::2],transform=ccrs.PlateCarree()
             ,headwidth=4,headlength=6,scale=15,width=0.002,color='darkgreen',alpha=0.95)
ax5.quiverkey(quiv1,0.95,1.05,0.3,"0.3",fontproperties={'size':11},labelpos='W')

shp_CN=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_country.shp")
ax5.add_geometries(shp_CN.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=0.8,facecolor='none')
shp_NWC=Reader("./data/NWC.shp")
ax5.add_geometries(shp_NWC.geometries(),crs=ccrs.PlateCarree(),edgecolor='darkred',linewidths=1.3,facecolor='none')

ax5.set_xticks(np.arange(-70,181,20))  #指定要显示的经纬度
ax5.set_yticks(np.arange(20,81,10))
ax5.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式
ax5.yaxis.set_major_formatter(LatitudeFormatter())
ax5.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax5.yaxis.set_minor_locator(mticker.MultipleLocator(5))

cbar_ax=fig.add_axes([0.170,0.037,0.600,0.010])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)
cbar.set_ticks(lines)

plt.savefig('./figure/Figure 3.pdf', bbox_inches='tight')
