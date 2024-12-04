import xarray as xr
import scipy.signal as scisig
from cartopy.util import add_cyclic_point
import matplotlib
matplotlib.use('Qt5Agg')

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14



#%%读取数据
f_Ctrl=xr.open_dataset('D:/data/xibei/ECHAM/ECHAM5/Ctrl/XB100ctrl_198001-199912.daily.nc')
f_Sen=xr.open_dataset('D:/data/xibei/ECHAM/ECHAM5/Sen/XB100sen_198001-199912.daily.nc')

z_Ctrl=f_Ctrl['geopoth'].sel(lev=20000,lat=slice(90,-10))/10   #除10单位变成位势米
z_Ctrl=z_Ctrl.sel(time=slice('1985','1999'))

z_Sen=f_Sen['geopoth'].sel(lev=20000,lat=slice(90,-10))/10    #除10单位变成位势米
z_Sen=z_Sen.sel(time=slice('1985','1999'))

lon,lat=z_Ctrl.lon,z_Ctrl.lat

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

z_Ctrl_bandpass=z_Ctrl.copy()
z_Ctrl_bandpass.values=scisig.filtfilt(taps, 1.0, z_Ctrl,axis=0)
z_Ctrl_bandpass_JJA=z_Ctrl_bandpass.sel(time=z_Ctrl_bandpass.time.dt.season=='JJA')
storm_track_Ctrl=z_Ctrl_bandpass_JJA.groupby('time.year').var(dim='time') #每年夏季的带通滤波方差
storm_track_Ctrl_mean=storm_track_Ctrl.mean(dim='year')

z_Sen_bandpass=z_Sen.copy()
z_Sen_bandpass.values=scisig.filtfilt(taps, 1.0, z_Sen,axis=0)
z_Sen_bandpass_JJA=z_Sen_bandpass.sel(time=z_Sen_bandpass.time.dt.season=='JJA')
storm_track_Sen=z_Sen_bandpass_JJA.groupby('time.year').var(dim='time') #每年夏季的带通滤波方差
storm_track_Sen_mean=storm_track_Sen.mean(dim='year')

storm_track_diff=storm_track_Sen_mean-storm_track_Ctrl_mean

storm_track_diff_cyclic,lon_cyclic=add_cyclic_point(storm_track_diff, coord=lon)

#%%保存数据

da1 = xr.DataArray(storm_track_diff_cyclic, coords=[('lat', lat.values), ('lon', lon_cyclic)], name='storm_track_diff')

ds = xr.Dataset({'storm_track_diff': da1})

ds.to_netcdf('./data/ECHAM_storm_track_diff.nc') #ECHAM5模式的瞬变涡旋响应