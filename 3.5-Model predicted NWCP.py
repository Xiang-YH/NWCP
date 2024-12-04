import numpy as np
import xarray as xr
import scipy.stats as st

def arealmean(var,lon,lat,lonrange,latrange):
    '''
    用于求多时次场变量某一区域的区域平均变量时间序列

    :param var: 多时次场变量
    :param lon: 变量的经度
    :param lat: 变量的纬度
    :param lonrange: 区域的经度范围
    :param latrange: 区域的纬度范围

    :return: 区域平均变量时间序列
    '''
    A=var[:,(lat>=latrange[0])&(lat<=latrange[1]),:]
    A=A[:,:,(lon>=lonrange[0])&(lon<=lonrange[1])]
    lat=lat[(lat>=latrange[0])&(lat<=latrange[1])]
    lon=lon[(lon>=lonrange[0])&(lon<=lonrange[1])]
    lon,lat=np.meshgrid(lon,lat)
    weight=np.cos(np.deg2rad(lat)) #纬度余弦加权
    weight[np.isnan(A[0,:,:])]=0 #缺测值格点权重为0
    A[np.isnan(A)]=0
    T=A.shape[0]
    ans=np.zeros([T])
    for t in range(T):
        ans[t]=np.average(A[t,:,:],weights=weight)
    return ans

def calc_MSSS(OBS,Simu):
    MSE=np.mean((Simu-OBS)**2)
    MSEc=np.mean((OBS-np.mean(OBS))**2)
    return 1-MSE/MSEc

def bzh(x):
    return (x-np.mean(x))/np.std(x)

#%%读取观测数据
mask_xb=np.load('./data/maskXB.npy')
fpre=xr.open_dataset('D:/data/xibei/CN05.1_Pre_1961_2021_monthly_1x1.nc')
pre=fpre.pre.sel(time=slice('1961','2021')) #读取1961-2021年
pre=pre.sel(time=pre.time.dt.season=='JJA') #筛选JJA
lon=pre.lon.values
lat=pre.lat.values
pre.values=pre.values*mask_xb #保留西北地区
pre_JJA=pre.coarsen(time=3).mean()  #求逐年夏季平均
pre_JJA_a=pre_JJA-pre_JJA.mean('time')
pre_hd=pre_JJA_a.rolling(time=4).mean() #进行4年滑动平均
pre_4=pre_hd[3:].values #4年滑动平均后前三个值是nan

P_OBS=np.load('./data/P_OBS_decadal.npy')

year=np.arange(1992,2017+1)

#%%读取模式数据
fmod=xr.open_dataset('./data/pre_lead_moving_average.nc')
pr_1_4_decadal=fmod['pre_1-4_decadal'].loc['1992':'2017'].values*86400 #单位转化成mm/day
pr_2_5_decadal=fmod['pre_2-5_decadal'].loc['1992':'2017'].values*86400 #单位转化成mm/day
pr_3_6_decadal=fmod['pre_3-6_decadal'].loc['1992':'2017'].values*86400 #单位转化成mm/day
pr_4_7_decadal=fmod['pre_4-7_decadal'].loc['1992':'2017'].values*86400 #单位转化成mm/day
pr_5_8_decadal=fmod['pre_5-8_decadal'].loc['1992':'2017'].values*86400 #单位转化成mm/day
pr_6_9_decadal=fmod['pre_6-9_decadal'].loc['1992':'2017'].values*86400 #单位转化成mm/day
pr_7_10_decadal=fmod['pre_7-10_decadal'].loc['1992':'2017'].values*86400 #单位转化成mm/day

pr_model=[pr_1_4_decadal,pr_2_5_decadal,pr_3_6_decadal,pr_4_7_decadal,pr_5_8_decadal,
           pr_6_9_decadal,pr_7_10_decadal]

lead_yeas=['1-4','2-5','3-6','4-7','5-8','6-9','7-10']

#%%计算模式降水异常
TCC=[]
MSSS=[]
OBS=arealmean(pre_4,lon,lat,[70,125],[25,60])[30:-2] #1992-2017年
for lead,pr_mod in zip(lead_yeas,pr_model):
    pr_mod_a=pr_mod-np.mean(pr_mod,0)
    X=arealmean(pr_mod_a,lon,lat,[70,125],[25,60])
    tcc=st.pearsonr(OBS,X)[0]
    TCC.append(tcc)
    msss=calc_MSSS(OBS,X)
    MSSS.append(msss)

np.savez('./data/direct_skill.npz',TCC=TCC,MSSS=MSSS) #模式直接预测西北降水的TCC和MSSS技巧