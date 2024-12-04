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


def multiregression(X,Y):
    '''
    计算并返回多元回归方程的回归系数、因子显著性，回归方程显著性
    :param X: [x1,x2,x3,...]其中xi是一维序列
    :param Y: 一维序列
    :return: K回归系数、P因子显著性，Fp回归方程显著性
    '''
    import statsmodels.api as sm
    X=np.vstack(X).T
    model=sm.OLS(Y,sm.add_constant(X)) #添加截距项
    res=model.fit()
    K,P,Fp=res.params,res.pvalues,res.f_pvalue
    return K,P,Fp


def corr_cef(_xx,_yy):
    """
    计算 _xx 对 _yy的相关系数(简单相关系数)
    :param _xx: 自变量, 一维
    :param _yy: 因变量, 一维
    :return: 相关系数值
    """
    return np.corrcoef(_xx,_yy)[0,1]


def partial_corr(_xx1,_yy,_xx2):
    """
    计算_xx1 对 _yy的偏相关系数
    :param _xx1: 自变量 1, 一维
    :param _xx2: 自变量 2, 一维
    :param _yy: 应变量, 一维
    :return: _xx1 对 _yy的偏相关系数值
    """
    numerator=corr_cef(_xx1,_yy)-(corr_cef(_yy,_xx2)*corr_cef(_xx1,_xx2))
    denominator=np.sqrt((1-corr_cef(_yy,_xx2)**2)*(1-corr_cef(_xx1,_xx2)**2))
    return numerator/denominator

#%% 读取观测数据
years=np.arange(1962,2019+1) #4年滑动平均后的模式预测序列的年份只到2017年

P_In=xr.DataArray(np.load('./data/P_In.npy'), dims='year', coords={'year': years})
P_F=xr.DataArray(np.load('./data/P_F.npy'), dims='year', coords={'year': years})
P=xr.DataArray(np.load('./data/P_OBS_decadal.npy'), dims='year', coords={'year': years})

predicters=np.load('./data/predicters.npz')
IO=xr.DataArray(predicters['IO'], dims='year', coords={'year': years})
PDO=xr.DataArray(predicters['PDO'], dims='year', coords={'year': years})
NAD=xr.DataArray(predicters['NAD'], dims='year', coords={'year': years})


IO_train=IO.loc[1962:2000].values
PDO_train=PDO.loc[1962:2000].values
NAD_train=NAD.loc[1962:2000].values
P_In_train=P_In.loc[1962:2000].values

IO_pred=IO.loc[2004:2020].values
PDO_pred=PDO.loc[2004:2020].values
NAD_pred=NAD.loc[2004:2020].values
P_In_pred=P_In.loc[2004:2020].values

#%% 蒙特卡洛检验
R_99=0.6182
R_95=0.4999
R_90=0.4305


#%%

print(f'r(NWCP-I, NAD)={st.pearsonr(P_In_train,NAD_train)[0]:.2f}\t',end='')
print(f'r(NWCP-I, NAD (PDO))={partial_corr(P_In_train,NAD_train,PDO_train):.2f}\t',end='')
print(f'r(NWCP-I, NAD (IO))={partial_corr(P_In_train,NAD_train,IO_train):.2f}\t')


print(f'r(NWCP-I, PDO)={st.pearsonr(P_In_train,PDO_train)[0]:.2f}\t',end='')
print(f'r(NWCP-I, PDO (NAD))={partial_corr(P_In_train,PDO_train,NAD_train):.2f}\t',end='')
print(f'r(NWCP-I, PDO (IO))={partial_corr(P_In_train,PDO_train,IO_train):.2f}\t')

print(f'r(NWCP-I, IO)={st.pearsonr(P_In_train,IO_train)[0]:.2f}\t',end='')
print(f'r(NWCP-I, IO (NAD))={partial_corr(P_In_train,IO_train,NAD_train):.2f}\t',end='')
print(f'r(NWCP-I, IO (PDO))={partial_corr(P_In_train,IO_train,PDO_train):.2f}\t')
