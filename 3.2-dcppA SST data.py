import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

#%%读取模式数据
path='D:/data/xibei/CMIP6/decadal/' #所有模式的数据都计算了多成员平均、JJA平均，并插值到统一分辨率
Models=['CMCC-CM2-SR5','MIROC6','MPI-ESM1-2-HR','NorCPM1']
SYear=['s'+str(year) for year in range(1960,2018+1)] #年代际预测数据的初始化时间

data=dict() #data的每个键是模式名
for model in Models: #读取每个模式的所有年代际预测数据
    data[model]=dict() #data中每个模式都对应一个字典，字典的键是年代际预测数据的初始化时间
    for syear in SYear:
        file_name=path+model+'./tos/tos_Omon_{}_dcppA-hindcast_{}_mme_remap_JJAmean_invertlat.nc'.format(model,syear)
        file=xr.open_dataset(file_name)
        tos=file['tos']
        data[model][syear]=tos

lon,lat=tos.lon,tos.lat
_,I,J=tos.shape

#%%求集合平均数据
data['MME']=dict()
for syear in tqdm(SYear): #求所有模式的集合平均
    MME_list=[]
    for model in Models:
        MME_list.append(data[model][syear].values)

    MME=xr.DataArray(np.nanmean(MME_list,axis=0),coords=data[model][syear].coords)
    data['MME'][syear]=MME

#%%拼接lead1-lead10年的十个时间序列
Tos_lead={} #用来储存十个不同提前期的海温场，key是提前期，value是海温场
for i in range(10): #初始化十个不同提前期的海温场的维度信息
    years=pd.to_datetime([str(year)+'-07-16' for year in np.arange(1961+i,2019+i+1)])
    # print(min(years),max(years))
    Tos_lead[str(i+1)]=xr.DataArray(np.zeros([len(SYear),I,J]),coords=[years,lat,lon],dims=["time","lat","lon"])

for ind,syear in enumerate(SYear): #syear是模式起报时间，ind是“当前模式数据要填充Tos_lead的第几个元素”
    tos_MME=data['MME'][syear]
    for i in range(10):
        Tos_lead[str(i+1)][ind,:,:]=tos_MME[i,:,:]

#%%进行滑动平均
def moving_average_4(array):
    array_hd=array.rolling(time=4).mean() #进行4年滑动平均
    array_decadal=array_hd[3:] #4年滑动平均后,前三个值是nan
    array_decadal['time']=array.time[1:-2] #滑动平均后的年份少了第一年和后两年
    return array_decadal

Tos_lead_ma={} #用来储存十个滑动平均后的不同提前期的海温场，key是提前期，value是4年滑动平均后的海温场
for i in range(10):
    Tos_lead_ma[str(i+1)]=moving_average_4(Tos_lead[str(i+1)])

#%%平均不同的提前期
for i in tqdm(range(7)): #1-4,2-5,3-6,4-7,5-8,6-9,7-10
    lead=str(i+1)+'-'+str(i+4)
    Tos_lead_ma[lead]=Tos_lead_ma[str(i+1)].copy(deep=True) #lead1-4和lead1的年份是一样的，以此类推
    Tos_lead_ma[lead][:]=np.nan #把数组里的值全部赋值为nan
    T_ma,_,_=Tos_lead_ma[lead].shape
    for t in range(T_ma):
        if t==0:
            Tos_lead_ma[lead][t,:,:]=Tos_lead_ma[str(i+1)][t,:,:]
        elif t==1:
            Tos_lead_ma[lead][t,:,:]=(Tos_lead_ma[str(i+1)][t,:,:]+Tos_lead_ma[str(i+2)][t-1,:,:])/2
        elif t==2:
            Tos_lead_ma[lead][t,:,:]=(Tos_lead_ma[str(i+1)][t,:,:]+Tos_lead_ma[str(i+2)][t-1,:,:]+Tos_lead_ma[str(i+3)][t-2,:,:])/3
        else:
            Tos_lead_ma[lead][t,:,:]=(Tos_lead_ma[str(i+1)][t,:,:]+Tos_lead_ma[str(i+2)][t-1,:,:]+Tos_lead_ma[str(i+3)][t-2,:,:]+Tos_lead_ma[str(i+4)][t-3,:,:])/4

#%%提取公共时段
for key in Tos_lead_ma.keys():
    Tos_lead_ma[key]=Tos_lead_ma[key].sel(time=slice('1968','2017'))

#%%保存数据
dataset = xr.Dataset()
for key in Tos_lead_ma.keys():
    dataset['tos_'+key+'_decadal']=Tos_lead_ma[key]

dataset.to_netcdf('./data/tos_lead_moving_average.nc') #模式不同提前期预测的海温场

