import numpy as np
import xarray as xr
import scipy.stats as st

def bzh(x):
    return (x-np.mean(x))/np.std(x)

def correlation(x,Y):
    T,I,J=Y.shape
    R=np.zeros([I,J])*np.nan
    for i in range(I):
        for j in range(J):
            if ~np.isnan(Y[:,i,j]).any():
                R[i,j]=st.pearsonr(x,Y[:,i,j])[0]
    return R

def calc_MSSS(OBS,Simu):
    MSE=np.mean((Simu-OBS)**2)
    MSEc=np.mean((OBS-np.mean(OBS))**2)
    return 1-MSE/MSEc


train_years=25 #----------------训练时段的长度--------------------


#%% 读取观测数据
years=np.arange(1962,2017+1) #4年滑动平均后的模式预测序列的年份只到2017年
P=xr.DataArray(np.load('./data/P_OBS_decadal.npy')[:-2], dims='year', coords={'year': years})

#%% 进行提前期平均
P_lead=xr.zeros_like(P) #第y年放的是y-3、y-2、y-1、y这4年的平均

for y in years:
    if y==1962:
        P_lead.loc[y]=P.loc[y]
    elif y==1963:
        P_lead.loc[y]=(P.loc[y-1]+P.loc[y])/2
    elif y==1964:
        P_lead.loc[y]=(P.loc[y-2]+P.loc[y-1]+P.loc[y])/3
    else:
        P_lead.loc[y]=(P.loc[y-3]+P.loc[y-2]+P.loc[y-1]+P.loc[y])/4

#%%计算不同提前期的技巧并画折线图
OBS=P.loc[1992:2017].values #滚动预测训练时段为25年时每个提前期预测序列的范围是1992-2017
TCC=[]
MSSS=[]

for i in range(7): #1-4,2-5,3-6,4-7,5-8,6-9,7-10
    lead=str(i+1)+'-'+str(i+4)
    # print(P_lead.loc[1992-i-1:2017-i-1].year)
    predict=P_lead.loc[1992-i-1:2017-i-1].values
    tcc=st.pearsonr(predict,OBS)[0]
    TCC.append(tcc)
    msss=calc_MSSS(OBS,predict)
    MSSS.append(msss)
    print('提前{:4}年  TCC={:.2f}, MSSS={:5.2f}'.format(lead,tcc,msss))

np.savez('./data/persistent_skill.npz',TCC=TCC,MSSS=MSSS) #持续预测的TCC和MSSS技巧