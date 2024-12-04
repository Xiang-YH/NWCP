import numpy as np
import xarray as xr
from metpy.units import units
import metpy.constants as metc
import scipy.signal as scisig
import matplotlib.pyplot as plt
import scipy.stats as st
import metpy.calc as mpcalc
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from scipy.ndimage import convolve
import time
import matplotlib
matplotlib.use('Qt5Agg')

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14

t0=time.time()

def draw_location(ax,color,lon1,lon2,lat1,lat2,lw):
    ax.plot([lon1,lon1],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
    ax.plot([lon2,lon2],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())
    ax.plot([lon1,lon2],[lat1,lat1],lw=lw,c=color,transform=ccrs.PlateCarree())
    ax.plot([lon1,lon2],[lat2,lat2],lw=lw,c=color,transform=ccrs.PlateCarree())



#%%读取数据
years=np.arange(1962,2020)
P_In=np.load('./data/P_In.npy')

predicters=np.load('./data/predicters.npz')
NAD=predicters['NAD']

p=70000*units('Pa')
lat_range=slice(0,80)
time_range=slice('1961','2021')

fu=xr.open_dataset('D:/data/xibei/ERA5-U_850_700_500-1961_2021-25_25-daily.nc')
u=fu['u'].sel(time=time_range,lat=lat_range,plev=p)
u_bar=u.sel(time=u.time.dt.season=='JJA').groupby('time.year').mean(dim='time')

fv=xr.open_dataset('D:/data/xibei/ERA5-V_850_700_500-1961_2021-25_25-daily.nc')
v=fv['v'].sel(time=time_range,lat=lat_range,plev=p)
v_bar=v.sel(time=v.time.dt.season=='JJA').groupby('time.year').mean(dim='time')

ft=xr.open_dataset('D:/data/xibei/ERA5-T_850_700_500-1961_2021-25_25-daily.nc')
t=ft['t'].sel(time=time_range,lat=lat_range,plev=p)
t_bar=t.sel(time=t.time.dt.season=='JJA').groupby('time.year').mean(dim='time')

fw=xr.open_dataset('D:/data/xibei/ERA5-W_850_700_500-1961_2021-25_25-daily.nc')
w=fw['w'].sel(time=time_range,lat=lat_range,plev=p)
w_bar=w.sel(time=w.time.dt.season=='JJA').groupby('time.year').mean(dim='time')

lon,lat=t.lon.values,t.lat.values
Lon,Lat=np.meshgrid(lon,lat)

print('读取数据完成',time.time()-t0)

P0=metc.P0
C_v=metc.Cv_d
C_p=metc.Cp_d
R=metc.Rd
g=metc.g


#%%带通滤波

# 计算滤波器参数
cutoff_period1 = 6  # 6天
cutoff_period2 = 2.5  # 2.5天
sampling_frequency = 1  # 每天1次
cutoff_frequency1 = 1 / cutoff_period1
cutoff_frequency2 = 1 / cutoff_period2

num_taps = 81  # 滤波器的抽头数（窗口长度）
window = 'lanczos'  # 窗口类型

# 使用firwin函数创建Lanczos窗口
taps = scisig.firwin(num_taps, [cutoff_frequency1,cutoff_frequency2], window=window, fs=sampling_frequency,pass_zero='bandpass')

u_bandpass=u.copy()
u_bandpass.values=scisig.filtfilt(taps, 1.0, u,axis=0)
u_bandpass_JJA=u_bandpass.sel(time=u_bandpass.time.dt.season=='JJA')*units('m/s')
print('u带通滤波完成',time.time()-t0)

v_bandpass=v.copy()
v_bandpass.values=scisig.filtfilt(taps, 1.0, v,axis=0)
v_bandpass_JJA=v_bandpass.sel(time=v_bandpass.time.dt.season=='JJA')*units('m/s')
print('v带通滤波完成',time.time()-t0)

t_bandpass=t.copy()
t_bandpass.values=scisig.filtfilt(taps, 1.0, t,axis=0)
t_bandpass_JJA=t_bandpass.sel(time=t_bandpass.time.dt.season=='JJA')*units('K')
print('t带通滤波完成',time.time()-t0)

w_bandpass=w.copy()
w_bandpass.values=scisig.filtfilt(taps, 1.0, w,axis=0)
w_bandpass_JJA=w_bandpass.sel(time=w_bandpass.time.dt.season=='JJA')*units('Pa/s')
print('w带通滤波完成',time.time()-t0)


#%% BTEC

btec1=(v_bandpass_JJA**2).groupby('time.year').mean(dim='time')-(u_bandpass_JJA**2).groupby('time.year').mean(dim='time')
btec2=mpcalc.first_derivative(u_bar,axis='lon')-mpcalc.first_derivative(v_bar,axis='lat')
btec3=-(u_bandpass_JJA*v_bandpass_JJA).groupby('time.year').mean(dim='time')
btec4=mpcalc.first_derivative(v_bar,axis='lon')+mpcalc.first_derivative(u_bar,axis='lat')

BTEC=P0/g*(btec1*btec2/2+btec3*btec4)
BTEC.data=BTEC.data.to(units('W/m**2'))


#%% BCEC1

t_alllev=ft['t'].sel(time=time_range,lat=lat_range)
t_alllev_mean=t_alllev.sel(time=t_alllev.time.dt.season=='JJA').mean(dim='time') #这一步之后xr.Dataarray没有units信息了
lev=t_alllev_mean.plev
theta_mean=mpcalc.potential_temperature(lev,t_alllev_mean*units('K'))
dthe_dp=mpcalc.first_derivative(theta_mean,axis='plev').sel(plev=p) #选定那一层的

bcec1_1=(u_bandpass_JJA*t_bandpass_JJA).groupby('time.year').mean(dim='time')
bcec1_2=mpcalc.first_derivative(t_bar,axis='lon')

bcec1_3=(v_bandpass_JJA*t_bandpass_JJA).groupby('time.year').mean(dim='time')
bcec1_4=mpcalc.first_derivative(t_bar,axis='lat')

C1=(P0/p)**(C_v/C_p)*R/g
C2=C1*(P0/p)**(R/C_p)/(-dthe_dp.mean())

BCEC1=-C2*(bcec1_1*bcec1_2+bcec1_3*bcec1_4)
BCEC1.data=BCEC1.data.to(units('W/m**2'))

#%% BECE2

BCEC2=-C1*(w_bandpass_JJA*t_bandpass_JJA).groupby('time.year').mean(dim='time')
BCEC2.data=BCEC2.data.to(units('W/m**2'))

#%% 四年滑动平均
BTEC_hd=BTEC.rolling(year=4).mean()  #进行4年滑动平均
BTEC_4=BTEC_hd[3:].values  #4年滑动平均后,前三个值是nan

BCEC1_hd=BCEC1.rolling(year=4).mean()  #进行4年滑动平均
BCEC1_4=BCEC1_hd[3:].values  #4年滑动平均后,前三个值是nan

BCEC2_hd=BCEC2.rolling(year=4).mean()  #进行4年滑动平均
BCEC2_4=BCEC2_hd[3:].values  #4年滑动平均后,前三个值是nan

print('四年滑动平均完成',time.time()-t0)


#%%提取训练时段
#训练时段1961-2002年，对应年代际序列的范围时1962-2000年
years_train=years[:-19]
print(years_train)

P_train=P_In[:-19]
NAD_train=NAD[:-19]

BTEC_4_train=BTEC_4[:-19]
BCEC1_4_train=BCEC1_4[:-19]
BCEC2_4_train=BCEC2_4[:-19]

#%% 蒙特卡洛检验
R_90=0.4305

#%% 计算预测因子同期相关

def regression(x,Y):
    T,I,J=Y.shape
    K=np.zeros([I,J])*np.nan
    R=np.zeros([I,J])*np.nan
    for i in range(I):
        for j in range(J):
            if ~np.isnan(Y[:,i,j]).any():
                K[i,j],_,R[i,j]=st.linregress(x,Y[:,i,j])[:3]
    return K,R

k_BTEC,r_BTEC=regression(NAD_train,BTEC_4_train)
k_BCEC1,r_BCEC1=regression(NAD_train,BCEC1_4_train)
k_BCEC2,r_BCEC2=regression(NAD_train,BCEC2_4_train)

print('计算回归场完成',time.time()-t0)

#%% 画图
fig=plt.figure(figsize=(12,3.5))
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 15
plt.subplots_adjust(left=0.05, bottom=0.20, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

lines=np.array([-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,-0.01,
                0.01,0.02,0.04,0.06,0.08,0.1,0.12]) *1e2  #单位10^-2 W/m2

#---------------------------------------------(1)--------------------------------------------------
ax1=plt.subplot(131,projection=ccrs.PlateCarree(0))
ax1.set_title('a',loc='left',fontweight='bold')
ax1.set_extent([-70,20,10,75],crs=ccrs.PlateCarree())
ax1.coastlines('50m',alpha=0.8,color='0.2')
c=ax1.contourf(lon,lat,k_BCEC1*1e2,lines,
               transform=ccrs.PlateCarree(),extend='both',alpha=0.8,cmap='RdBu_r')
ax1.scatter(Lon[np.abs(r_BCEC1)>R_90],Lat[np.abs(r_BCEC1)>R_90],marker='o', ############ R90
            color='w',s=4,alpha=0.4,transform=ccrs.PlateCarree())
ax1.set_xticks(np.arange(-70,21,15))  #指定要显示的经纬度
ax1.set_yticks(np.arange(15,76,15))
ax1.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(5))

#---------------------------------------------(b)--------------------------------------------------
ax2=plt.subplot(132,projection=ccrs.PlateCarree(0))
ax2.set_title('b',loc='left',fontweight='bold')
ax2.set_extent([-70,20,10,75],crs=ccrs.PlateCarree())
ax2.coastlines('50m',alpha=0.8,color='0.2')

kernel = np.ones((3, 3))
kernel /= kernel.sum()
k_BCEC2_somth = convolve(k_BCEC2, kernel)

c=ax2.contourf(lon,lat,k_BCEC2_somth*1e2,lines,
               transform=ccrs.PlateCarree(),extend='both',alpha=0.8,cmap='RdBu_r')
ax2.scatter(Lon[np.abs(r_BCEC2)>R_90],Lat[np.abs(r_BCEC2)>R_90],marker='o', ############ R90
            color='w',s=4,alpha=0.4,transform=ccrs.PlateCarree())
ax2.set_xticks(np.arange(-70,21,15))  #指定要显示的经纬度
ax2.set_yticks(np.arange(15,76,15))
ax2.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax2.yaxis.set_major_formatter(LatitudeFormatter())
ax2.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(5))

cbar_ax=fig.add_axes([0.15,0.07,0.7,0.03])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)
cbar.set_ticks(lines)
cbar.ax.tick_params(labelsize=10)

#---------------------------------------------(c)--------------------------------------------------
ax3=plt.subplot(133,projection=ccrs.PlateCarree(0))
ax3.set_title('c',loc='left',fontweight='bold')
ax3.set_extent([-70,20,10,75],crs=ccrs.PlateCarree())
ax3.coastlines('50m',alpha=0.8,color='0.2')
c=ax3.contourf(lon,lat,k_BTEC*1e2,lines,
               transform=ccrs.PlateCarree(),extend='both',alpha=0.8,cmap='RdBu_r')
ax3.scatter(Lon[np.abs(r_BTEC)>R_90],Lat[np.abs(r_BTEC)>R_90],marker='o', ############ R90
            color='w',s=4,alpha=0.4,transform=ccrs.PlateCarree())
ax3.set_xticks(np.arange(-70,21,15))  #指定要显示的经纬度
ax3.set_yticks(np.arange(15,76,15))
ax3.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax3.yaxis.set_major_formatter(LatitudeFormatter())
ax3.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax3.yaxis.set_minor_locator(mticker.MultipleLocator(5))

plt.savefig('./figure/Supplementary Figure 3.pdf', bbox_inches='tight')