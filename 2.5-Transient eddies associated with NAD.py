import numpy as np
import xarray as xr
import scipy.signal as scisig
import scipy.stats as st
import matplotlib
matplotlib.use('Qt5Agg')

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14



#%%读取数据
years=np.arange(1962,2020)
P_In=np.load('./data/P_In.npy')


predicters=np.load('./data/predicters.npz')
NAD=predicters['NAD']

f=xr.open_dataset('D:/data/xibei/ERA5-Z200-1961_2021-25_25-daily.nc')
z=f['z'].sel(time=slice('1961','2021'),lat=slice(-15,90))/9.8/10    #位势/g=位势高度（位势米），再除10单位变成位势十米
lon,lat=z.lon,z.lat
Lon,Lat=np.meshgrid(lon,lat)

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

z_bandpass=z.copy()
z_bandpass.values=scisig.filtfilt(taps, 1.0, z,axis=0)
z_bandpass_JJA=z_bandpass.sel(time=z_bandpass.time.dt.season=='JJA')

storm_track=z_bandpass_JJA.groupby('time.year').var(dim='time') #每年夏季的带通滤波方差

storm_track_mean_state=storm_track.mean(dim='year')

#%% 四年滑动平均
storm_track_hd=storm_track.rolling(year=4).mean()  #进行4年滑动平均
storm_track_4=storm_track_hd[3:].values  #4年滑动平均后,前三个值是nan


#%%提取训练时段
#训练时段1961-2002年，对应年代际序列的范围时1962-2000年
years_train=years[:-19]
print(years_train)

P_train=P_In[:-19]

storm_track_4_train=storm_track_4[:-19]

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

k_storm_track,r_storm_track=regression(NAD_train,storm_track_4_train)


#%%保存数据
da1 = xr.DataArray(storm_track_mean_state, coords=[('lat', lat.values), ('lon', lon.values)], name='storm_track_mean_state')
da2 = xr.DataArray(k_storm_track, coords=[('lat', lat.values), ('lon', lon.values)], name='k_storm_track')
da3 = xr.DataArray(r_storm_track, coords=[('lat', lat.values), ('lon', lon.values)], name='r_storm_track')

ds = xr.Dataset({'storm_track_mean_state': da1, 'k_storm_track': da2, 'r_storm_track': da3})

ds.to_netcdf('./data/NAD_storm_track.nc') #风暴轴平均态和回归至NAD的200hPa瞬变涡旋强度
