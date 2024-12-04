import numpy as np
import xarray as xr
import scipy.stats as st
import matplotlib.pyplot as plt

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

P_In=xr.DataArray(np.load('./data/P_In.npy')[:-2], dims='year', coords={'year': years})
P_F=xr.DataArray(np.load('./data/P_F.npy')[:-2], dims='year', coords={'year': years})
P=xr.DataArray(np.load('./data/P_OBS_decadal.npy')[:-2], dims='year', coords={'year': years})

predicters=np.load('./data/predicters.npz')
NAD=xr.DataArray(predicters['NAD'][:-2], dims='year', coords={'year': years})


#%%滚动预测
Perfect_lead={} #用于储存完美预测的不同提前期的预测结果，key是提前期，value是预测结果序列
for i in range(10):
    Perfect_lead[str(i+1)]=[] #初始化每个提前期的列表，用于后面append

for y in range(1961,2016-train_years+2): #最晚只能获得2017年的模式预测海温，所以训练时段最多到2016年

    #----------由训练时段得到回归方程---------
    start_year,end_year=y,y+train_years-1 #训练时段终止年end=start+L-1
    print(f'训练时段：{start_year}-{end_year}',end='  ')
    P_In_train=P_In.loc[start_year+1:end_year-2] #年代际序列的开始年=start+1，终止年=end-2 (loc的特殊用法end-2是取得到的)
    NAD_train=NAD.loc[start_year+1:end_year-2]
    print(f'年代际序列：{P_In_train.year.values.min()}-{P_In_train.year.values.max()}',end='  ')
    slope,intercept,r_value,_,_=st.linregress(NAD_train.values,P_In_train.values)
    print('k={:5.4f}, b={:7.4f}, r={:.2f}'.format(slope,intercept,r_value),end='  ')

    #----------把观测中的预测因子值代入回归方程-----------
    NAD_predict=NAD.loc[end_year+1:end_year+10] #预测时段是训练时段结束后的十年
    P_F_predict=P_F.loc[end_year+1:end_year+10]
    print(f'预测时段：{NAD_predict.year.values.min()}-{NAD_predict.year.values.max()}')
    P_predict=slope*NAD_predict.values+intercept+P_F_predict #预测的降水异常=预测的内部变率+外强迫
    for i,predict in enumerate(P_predict):
        Perfect_lead[str(i+1)].append(predict)

for i in range(10):
    Perfect_lead[str(i+1)]=np.array(Perfect_lead[str(i+1)]) #把列表转化为数组


#%% 平均不同的提前期
for i in range(7): #1-4,2-5,3-6,4-7,5-8,6-9,7-10
    lead=str(i+1)+'-'+str(i+4)
    Perfect_lead[lead]=Perfect_lead[str(i+1)].copy() #lead1-4和lead1的年份是一样的，以此类推
    T=len(Perfect_lead[lead])
    for t in range(T):
        if t==0:
            Perfect_lead[lead][t]=Perfect_lead[str(i+1)][t]
        elif t==1:
            Perfect_lead[lead][t]=(Perfect_lead[str(i+1)][t]+Perfect_lead[str(i+2)][t-1])/2
        elif t==2:
            Perfect_lead[lead][t]=(Perfect_lead[str(i+1)][t]+Perfect_lead[str(i+2)][t-1]+Perfect_lead[str(i+3)][t-2])/3
        else:
            Perfect_lead[lead][t]=(Perfect_lead[str(i+1)][t]+Perfect_lead[str(i+2)][t-1]+Perfect_lead[str(i+3)][t-2]+Perfect_lead[str(i+4)][t-3])/4

#%%提取公共时段
T_min=len(Perfect_lead['7-10']) #提前7-10年的序列长度是最短的
years_predict=np.arange(2017-T_min+1,2017+1)
print('不同提前期的序列长度：',T_min)
print('不同提前期的序列年份：',years_predict)
for i in range(7): #1-4,2-5,3-6,4-7,5-8,6-9,7-10
    lead=str(i+1)+'-'+str(i+4)
    Perfect_lead[lead]=Perfect_lead[lead][-T_min:] #只保留后几个元素

OBS=P.loc[2017-T_min+1:2017].values

#%%计算不同提前期的技巧
TCC=[]
MSSS=[]
for i in range(7): #1-4,2-5,3-6,4-7,5-8,6-9,7-10
    lead=str(i+1)+'-'+str(i+4)
    tcc=st.pearsonr(Perfect_lead[lead],OBS)[0]
    TCC.append(tcc)
    msss=calc_MSSS(OBS,Perfect_lead[lead])
    MSSS.append(msss)
    print('提前{:4}年  TCC={:.4f}, MSSS={:5.4f}'.format(lead,tcc,msss))

np.savez('./data/perfect_skill.npz',TCC=TCC,MSSS=MSSS) #完美预测的TCC和MSSS技巧