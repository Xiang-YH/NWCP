import numpy as np
import xarray as xr
from metpy.units import units
import metpy.constants as metc
import scipy.signal as scisig
import scipy.stats as st
import metpy.calc as mpcalc
from tqdm import tqdm
import time
import matplotlib
matplotlib.use('Qt5Agg')

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14

t0=time.time()

#%%读取数据
years=np.arange(1962,2020)
P_In=np.load('./data/P_In.npy')

predicters=np.load('./data/predicters.npz')
NAD=predicters['NAD']


fu=xr.open_dataset('D:/data/xibei/ERA5-U_300_250_200-1961_2021-25_25-daily.nc')
u=fu['u'].sel(time=slice('1961','2021'),lat=slice(-15,90),plev=20000)

fv=xr.open_dataset('D:/data/xibei/ERA5-V_300_250_200-1961_2021-25_25-daily.nc')
v=fv['v'].sel(time=slice('1961','2021'),lat=slice(-15,90),plev=20000)

fvort=xr.open_dataset('D:/data/xibei/ERA5-vort_300_250_200-1961_2021-25_25-daily.nc')
vort=fvort['vo'].sel(time=slice('1961','2021'),lat=slice(-15,90),plev=20000)

lon,lat=u.lon,u.lat
dx,dy=mpcalc.lat_lon_grid_deltas(lon,lat)
Dx = dx[None, :]
Dy = dy[None, :]
T,I,J=u.shape

print('读取完成',time.time()-t0)


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

vort_bandpass=v.copy()
vort_bandpass.values=scisig.filtfilt(taps, 1.0, vort,axis=0)
vort_bandpass_JJA=vort_bandpass.sel(time=vort_bandpass.time.dt.season=='JJA')*units('1/s')
print('vort带通滤波完成',time.time()-t0)

#%%计算夏季逐日涡度输送
u_vort_daily=u_bandpass_JJA*vort_bandpass_JJA
v_vort_daily=v_bandpass_JJA*vort_bandpass_JJA

print('计算夏季逐日涡度输送完成',time.time()-t0)

#%%计算夏季平均涡度输送散度
u_vort=u_vort_daily.groupby('time.year').mean(dim='time')
v_vort=v_vort_daily.groupby('time.year').mean(dim='time')

vort_flux_div=-mpcalc.divergence(u_vort.values,v_vort.values,dx=Dx,dy=Dy) #负的夏季平均瞬变涡旋通量散度

print('计算夏季平均涡度输送散度完成',time.time()-t0)


#%%求逆拉普拉斯得到位势倾向

def Inverse_Laplace(A,dx,dy):
    R=np.zeros_like(A)
    Z=np.zeros_like(A)
    dx=np.array(dx)
    dy=np.array(dy)
    A=np.array(A)
    dx=np.pad(dx,((0,0),(0,1)),mode='edge')  #dx数组比u、v少一列，扩展至一样的尺寸
    dy=np.pad(dy,((0,1),(0,0)),mode='edge')  #dy数组比u、v少一行，扩展至一样的尺寸

    N=5000
    for n in tqdm(range(N),desc="解逆拉普拉斯"):

        R[:,1:I-1,1:J-1]=(Z[:,2:I,1:J-1]+Z[:,0:I-2,1:J-1])/(dy[1:I-1,1:J-1]**2)+\
                           (Z[:,1:I-1,2:J]+Z[:,1:I-1,0:J-2])/(dx[1:I-1,1:J-1]**2)-\
                           2*(1/dx[1:I-1,1:J-1]**2+1/dy[1:I-1,1:J-1]**2)*Z[:,1:I-1,1:J-1]-\
                           A[:,1:I-1,1:J-1]

        R[:,1:I-1,0]=(Z[:,2:I,0]+Z[:,0:I-2,0])/(dy[1:I-1,0]**2)+\
                       (Z[:,1:I-1,1]+Z[:,1:I-1,-1])/(dx[1:I-1,0]**2)-\
                       2*(1/dx[1:I-1,0]**2+1/dy[1:I-1,0]**2)*Z[:,1:I-1,0]-\
                       A[:,1:I-1,0]

        R[:,1:I-1,-1]=(Z[:,2:I,-1]+Z[:,0:I-2,-1])/(dy[1:I-1,-1]**2)+\
                        (Z[:,1:I-1,0]+Z[:,1:I-1,-2])/(dx[1:I-1,-1]**2)-\
                        2*(1/dx[1:I-1,-1]**2+1/dy[1:I-1,-1]**2)*Z[:,1:I-1,-1]-\
                        A[:,1:I-1,-1]

        Z=Z+R/(2*(1/dx**2+1/dy**2))

    return Z

Lon,Lat=np.meshgrid(lon,lat)
f=mpcalc.coriolis_parameter(np.radians(Lat)) #地转涡度
g=metc.g #重力加速度

A=u_vort.copy()
A.data=Inverse_Laplace(vort_flux_div,dx,dy)*units('m**2/s**2') #计算逆拉普拉斯算子项
dzdt=f/g*A
dzdt.data=dzdt.data.to(units('m/day')) #跟doi:10.1175/jcli-d-21-0705.1一样单位变成gpm/day

#%% 四年滑动平均
dzdt_hd=dzdt.rolling(year=4).mean()  #进行4年滑动平均
dzdt_4=dzdt_hd[3:].values  #4年滑动平均后,前三个值是nan

print('四年滑动平均完成',time.time()-t0)


#%%提取训练时段
#训练时段1961-2002年，对应年代际序列的范围时1962-2000年
years_train=years[:-19]
print(years_train)

P_train=P_In[:-19]

dzdt_4_train=dzdt_4[:-19]

NAD_train=NAD[:-19]


#%% 蒙特卡洛检验
R_95=0.4999
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

k_dzdt,r_dzdt=regression(NAD_train,dzdt_4_train)

print('计算回归场完成',time.time()-t0)

#
#%%保存数据
da1 = xr.DataArray(k_dzdt, coords=[('lat', lat.values), ('lon', lon.values)], name='k_dzdt')
da2 = xr.DataArray(r_dzdt, coords=[('lat', lat.values), ('lon', lon.values)], name='r_dzdt')

ds = xr.Dataset({'k_dzdt': da1,'r_dzdt': da2})

ds.to_netcdf('./data/NAD_dzdt.nc') #回归至NAD的瞬变涡旋引起的位势高度倾向
