import numpy as np
import xarray as xr
import scipy.stats as st

def multiregression(X,Y):
    '''
    计算并返回多元回归方程的回归系数、因子显著性，回归方程显著性
    :param X: [x1,x2,x3,...]其中xi是一维序列
    :param Y: 三维数组
    :return: K回归系数、P因子显著性，Fp回归方程显著性
    '''
    import statsmodels.api as sm
    X=np.vstack(X).T
    model=sm.OLS(Y,sm.add_constant(X)) #添加截距项
    # model=sm.OLS(Y,X)  #不添加截距项
    res=model.fit()
    K,P,Fp=res.params,res.pvalues,res.f_pvalue
    return K,P,Fp



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
# print(np.max(P-(P_In+P_F))) #验证一下两者是等价的

predicters=np.load('./data/predicters.npz')
NAD=xr.DataArray(predicters['NAD'][:-2], dims='year', coords={'year': years})

#%%读取模式年代际预测数据
fmod=xr.open_dataset('./data/NAD_lead_moving_average.nc')
result=np.empty((len(fmod.time), 10))  #创建一个数组用于把10列不同提前期的模式数据拼在一起
result[:]=np.nan  # 先将所有元素设为nan

for i in range(10):
    lead_data=fmod['NAD_'+str(i+1)+'_decadal'].values
    # lead_data=fmod['NAD_'+str(i+1)+'_decadal'].time.dt.year  #可以验证错位排放是不是弄对了
    result[:-i if i!=0 else None,i]=lead_data[i:]  #leadi的序列向上移动i格后填入到第i列

years_init=np.arange(1967,2016+1) #十个lead时间序列的时间都是1968-2017，对应的模式起报时间是1967-2016
tos_mod=xr.DataArray(result, dims=('init_year', 'lead'), coords={'init_year': years_init, 'lead': range(1, 11)})


#%%滚动预测
Hybrid_lead={} #用于储存完美预测的不同提前期的预测结果，key是提前期，value是预测结果序列
for i in range(10):
    Hybrid_lead[str(i+1)]=[] #初始化每个提前期的列表，用于后面append

for y in range(1961,2016-train_years+2): #最晚只能获得2017年的模式预测海温，所以训练时段最多到2016年

    #----------由训练时段得到回归方程---------
    start_year,end_year=y,y+train_years-1 #训练时段终止年end=start+L-1
    print(f'训练时段：{start_year}-{end_year}',end='  ')
    P_In_train=P_In.loc[start_year+1:end_year-2] #年代际序列的开始年=start+1，终止年=end-2 (loc的特殊用法end-2是取得到的)
    NAD_train=NAD.loc[start_year+1:end_year-2]
    print(f'年代际序列：{P_In_train.year.values.min()}-{P_In_train.year.values.max()}',end='  ')
    slope,intercept,r_value,_,_=st.linregress(NAD_train.values,P_In_train.values)
    print('k={:5.4f}, b={:7.4f}, r={:.2f}'.format(slope,intercept,r_value),end='  ')

    #----------把动力模式中的预测因子值代入回归方程-----------
    NAD_predict=tos_mod.sel(init_year=end_year) #模式起报年份=训练期终止年份
    NAD_predict=np.array([value for value in NAD_predict.values if not np.isnan(value)]) #只保留不为nan的部分
    # print(NAD_predict)
    P_F_predict=P_F.loc[end_year+1:end_year+10]
    print(f'预测时段：{P_F_predict.year.values.min()}-{P_F_predict.year.values.max()}')
    P_predict=slope*NAD_predict+intercept+P_F_predict #预测的降水异常=预测的内部变率+外强迫
    for i,predict in enumerate(P_predict):
        Hybrid_lead[str(i+1)].append(predict)

for i in range(10):
    Hybrid_lead[str(i+1)]=np.array(Hybrid_lead[str(i+1)]) #把列表转化为数组


#%% 平均不同的提前期
for i in range(7): #1-4,2-5,3-6,4-7,5-8,6-9,7-10
    lead=str(i+1)+'-'+str(i+4)
    Hybrid_lead[lead]=Hybrid_lead[str(i+1)].copy() #lead1-4和lead1的年份是一样的，以此类推
    T=len(Hybrid_lead[lead])
    for t in range(T):
        if t==0:
            Hybrid_lead[lead][t]=Hybrid_lead[str(i+1)][t]
        elif t==1:
            Hybrid_lead[lead][t]=(Hybrid_lead[str(i+1)][t]+Hybrid_lead[str(i+2)][t-1])/2
        elif t==2:
            Hybrid_lead[lead][t]=(Hybrid_lead[str(i+1)][t]+Hybrid_lead[str(i+2)][t-1]+Hybrid_lead[str(i+3)][t-2])/3
        else:
            Hybrid_lead[lead][t]=(Hybrid_lead[str(i+1)][t]+Hybrid_lead[str(i+2)][t-1]+Hybrid_lead[str(i+3)][t-2]+Hybrid_lead[str(i+4)][t-3])/4

#%%提取公共时段
T_min=len(Hybrid_lead['7-10']) #提前7-10年的序列长度是最短的
years_predict=np.arange(2017-T_min+1,2017+1)
print('不同提前期的序列长度：',T_min)
print('不同提前期的序列年份：',years_predict)
for i in range(7): #1-4,2-5,3-6,4-7,5-8,6-9,7-10
    lead=str(i+1)+'-'+str(i+4)
    Hybrid_lead[lead]=Hybrid_lead[lead][-T_min:] #只保留后几个元素

OBS=P.loc[2017-T_min+1:2017].values

#%%计算不同提前期的技巧
TCC=[]
MSSS=[]
for i in range(7): #1-4,2-5,3-6,4-7,5-8,6-9,7-10
    lead=str(i+1)+'-'+str(i+4)
    tcc=st.pearsonr(Hybrid_lead[lead],OBS)[0]
    TCC.append(tcc)
    msss=calc_MSSS(OBS,Hybrid_lead[lead])
    MSSS.append(msss)
    print('提前{:4}年  TCC={:.4f}, MSSS={:5.4f}'.format(lead,tcc,msss))

np.savez('./data/hybrid_skill.npz',TCC=TCC,MSSS=MSSS) #动力统计结合预测的TCC和MSSS技巧