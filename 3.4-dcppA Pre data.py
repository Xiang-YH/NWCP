import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

mask_xb=np.load('./data/maskXB.npy')

def calc_MSSS(OBS,Simu):
    MSE=np.mean((Simu-OBS)**2)
    MSEc=np.mean((OBS-np.mean(OBS))**2)
    return 1-MSE/MSEc

#%%读取模式数据
path='D:/data/xibei/CMIP6/decadal/' #所有模式的数据都计算了多成员平均、JJA平均，并插值到统一分辨率
Models=['CMCC-CM2-SR5','MIROC6','MPI-ESM1-2-HR','NorCPM1']
SYear=['s'+str(year) for year in range(1960,2018+1)] #年代际预测数据的初始化时间

data=dict() #data的每个键是模式名
for model in Models: #读取每个模式的所有年代际预测数据
    data[model]=dict() #data中每个模式都对应一个字典，字典的键是年代际预测数据的初始化时间
    for syear in SYear:
        file_name=path+model+'./pr/pr_Amon_{}_dcppA-hindcast_{}_mme_remap_JJAmean.nc'.format(model,syear)
        file=xr.open_dataset(file_name)
        pr=file['pr']*mask_xb
        data[model][syear]=pr

lon,lat=pr.lon,pr.lat
_,I,J=pr.shape

#%%求集合平均数据
data['MME']=dict()
for syear in tqdm(SYear): #求所有模式的集合平均
    MME_list=[]
    for model in Models:
        MME_list.append(data[model][syear].values)

    MME=xr.DataArray(np.nanmean(MME_list,axis=0),coords=data[model][syear].coords)
    data['MME'][syear]=MME

#%%拼接lead1-lead10年的十个时间序列
Pre_lead={} #用来储存十个不同提前期的降水场，key是提前期，value是降水场
for i in range(10): #初始化十个不同提前期的降水场的维度信息
    years=pd.to_datetime([str(year)+'-07-16' for year in np.arange(1961+i                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ,2019+i+1)])
    # print(min(years),max(years))
    Pre_lead[str(i+1)]=xr.DataArray(np.zeros([len(SYear),I,J]),coords=[years,lat,lon],dims=["time","lat","lon"])

for ind,syear in enumerate(SYear): #syear是模式起报时间，ind是“当前模式数据要填充Pre_lead的第几个元素”
    pre_MME=data['MME'][syear]
    for i in range(10):
        Pre_lead[str(i+1)][ind,:,:]=pre_MME[i,:,:]

#%%进行滑动平均
def moving_average_4(array):
    array_hd=array.rolling(time=4).mean() #进行4年滑动平均
    array_decadal=array_hd[3:] #4年滑动平均后,前三个值是nan
    array_decadal['time']=array.time[1:-2] #滑动平均后的年份少了第一年和后两年
    return array_decadal

Pre_lead_ma={} #用来储存十个滑动平均后的不同提前期的降水场，key是提前期，value是4年滑动平均后的降水场
for i in range(10):
    Pre_lead_ma[str(i+1)]=moving_average_4(Pre_lead[str(i+1)])
    
#%%平均不同的提前期
for i in tqdm(range(7)): #1-4,2-5,3-6,4-7,5-8,6-9,7-10
    lead=str(i+1)+'-'+str(i+4)
    Pre_lead_ma[lead]=Pre_lead_ma[str(i+1)].copy(deep=True) #lead1-4和lead1的年份是一样的，以此类推
    Pre_lead_ma[lead][:]=np.nan #把数组里的值全部赋值为nan
    T_ma,_,_=Pre_lead_ma[lead].shape
    for t in range(T_ma):
        if t==0:
            Pre_lead_ma[lead][t,:,:]=Pre_lead_ma[str(i+1)][t,:,:]
        elif t==1:
            Pre_lead_ma[lead][t,:,:]=(Pre_lead_ma[str(i+1)][t,:,:]+Pre_lead_ma[str(i+2)][t-1,:,:])/2
        elif t==2:
            Pre_lead_ma[lead][t,:,:]=(Pre_lead_ma[str(i+1)][t,:,:]+Pre_lead_ma[str(i+2)][t-1,:,:]+Pre_lead_ma[str(i+3)][t-2,:,:])/3
        else:
            Pre_lead_ma[lead][t,:,:]=(Pre_lead_ma[str(i+1)][t,:,:]+Pre_lead_ma[str(i+2)][t-1,:,:]+Pre_lead_ma[str(i+3)][t-2,:,:]+Pre_lead_ma[str(i+4)][t-3,:,:])/4

#%%提取公共时段
for key in Pre_lead_ma.keys():
    Pre_lead_ma[key]=Pre_lead_ma[key].sel(time=slice('1968','2017'))

#%%保存数据
dataset = xr.Dataset()
for key in Pre_lead_ma.keys():
    dataset['pre_'+key+'_decadal']=Pre_lead_ma[key]

dataset.attrs['warning']='单位是kg m-2 s-1'
dataset.to_netcdf('./data/pre_lead_moving_average.nc') #模式不同提前期预测的西北地区降水场
