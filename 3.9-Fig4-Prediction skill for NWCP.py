import numpy as np
import xarray as xr
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import ticker

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
years=np.arange(1962,2019+1)

P_In=xr.DataArray(np.load('./data/P_In.npy'), dims='year', coords={'year': years})
P_F=xr.DataArray(np.load('./data/P_F.npy'), dims='year', coords={'year': years})
P=xr.DataArray(np.load('./data/P_OBS_decadal.npy'), dims='year', coords={'year': years})

predicters=np.load('./data/predicters.npz')
NAD=xr.DataArray(predicters['NAD'], dims='year', coords={'year': years})


#%%每次预测10年的滚动预测
Predict_In=[]
Predict_P=[]

for y in range(1961,2018-train_years+2,10):

    #----------由训练时段得到回归方程---------
    start_year,end_year=y,y+train_years-1 #训练时段终止年end=start+L-1
    print(f'训练时段：{start_year}-{end_year}',end='  ')
    P_In_train=P_In.loc[start_year+1:end_year-2] #年代际序列的开始年=start+1，终止年=end-2 (loc的特殊用法end-2是取得到的)
    NAD_train=NAD.loc[start_year+1:end_year-2]
    print(f'年代际序列：{P_In_train.year.values.min()}-{P_In_train.year.values.max()}',end='  ')
    slope,intercept,r_value,_,_=st.linregress(NAD_train.values,P_In_train.values)
    print('k={:5.4f}, b={:7.4f}, r={:.2f}'.format(slope,intercept,r_value),end='  ')

    #----------把观测中的预测因子值代入回归方程-----------
    NAD_predict=NAD.loc[end_year+1:end_year+10] #预测时段是训练时段结束后的第一年
    P_F_predict=P_F.loc[end_year+1:end_year+10]
    print(f'预测时段：{NAD_predict.year.values.min()}-{NAD_predict.year.values.max()}')
    Predict_In.append(slope*NAD_predict.values+intercept)
    Predict_P.append(slope*NAD_predict.values+intercept+P_F_predict) #预测的降水异常=预测的内部变率+外强迫

Predict_In=np.concatenate(Predict_In)
Predict_P=np.concatenate(Predict_P)

#%%每次预测10年的预测技巧
OBS_In=P_In.loc[1961+train_years:2019].values
OBS_P=P.loc[1961+train_years:2019].values

years_pred=np.arange(1961+train_years,2019+1)

tcc_In_1=st.pearsonr(OBS_In,Predict_In)[0] #整个预测序列的TCC
msss_In_1=calc_MSSS(OBS_In,Predict_In) #整个预测序列的MSSS

tcc_In_2=st.pearsonr(OBS_In[years_pred>=2004],Predict_In[years_pred>=2004])[0] #独立时段预测序列的TCC
msss_In_2=calc_MSSS(OBS_In[years_pred>=2004],Predict_In[years_pred>=2004]) #独立时段预测序列的MSSS

tcc_P_1=st.pearsonr(OBS_P,Predict_P)[0] #整个预测序列的TCC
msss_P_1=calc_MSSS(OBS_P,Predict_P) #整个预测序列的MSSS

tcc_P_2=st.pearsonr(OBS_P[years_pred>=2004],Predict_P[years_pred>=2004])[0] #独立时段预测序列的TCC
msss_P_2=calc_MSSS(OBS_P[years_pred>=2004],Predict_P[years_pred>=2004]) #独立时段预测序列的MSSS

#%% 读取预测技巧
direct_skill=np.load('./data/direct_skill.npz') #模式直接预测技巧
direct_TCC,direct_MSSS=direct_skill['TCC'],direct_skill['MSSS']

perfect_skill=np.load('./data/perfect_skill.npz') #完美预测技巧
perfect_TCC,perfect_MSSS=perfect_skill['TCC'],perfect_skill['MSSS']

hybrid_skill=np.load('./data/hybrid_skill.npz') #动力统计结合预测技巧
hybrid_TCC,hybrid_MSSS=hybrid_skill['TCC'],hybrid_skill['MSSS']

persistent_skill=np.load('./data/persistent_skill.npz') #持续预测技巧
persistent_TCC,persistent_MSSS=persistent_skill['TCC'],persistent_skill['MSSS']

#%%画图
fig=plt.figure(figsize=(9,7.5))
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 15
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
lw=1.5
s=100

#---------------------------------------------(a)--------------------------------------------------
ax1=fig.add_axes([0.099,0.613,0.818,0.319])
years_predict=np.arange(1961+train_years,2019+1)
cmap = plt.get_cmap('Reds_r')  # 这里使用的是红色的颜色映射
colors = [] # 计算每个点的颜色值
color_loc=np.linspace(0.1, 0.9, 10)
for i in range(3):
    for j in color_loc:  # 每组10个点
        colors.append(cmap(j))
for j in color_loc[:4]:
    colors.append(cmap(j))

# 生成额外的图例
legend_elements = []
for i, color in enumerate(color_loc):
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'{i+1}yr',
                                      markerfacecolor=cmap(color), markersize=8))

first_legend=plt.legend(handles=legend_elements, title="Lead", frameon=False, bbox_to_anchor=(1.05, 0.5), loc='center') #生成提前期的图例
ax0 = plt.gca()  # 获取当前轴
ax0.add_artist(first_legend)  # 添加第一个图例到轴上
ax1.axhline(0,color='grey')
ax1.set_title('a',loc='left',fontweight='bold')
ax1.plot(years,P,label='Observation',c='k',lw=2.5,zorder=1)
ax1.plot(years_predict,Predict_P,label='Perfect prediction',c='r',lw=2.5,zorder=2)
ax1.scatter(years_predict,Predict_P,color=colors,s=40,zorder=3)
ax1.set_ylim(-0.2,0.2)
ax1.set_ylabel('Intensity (mm/day)')
ax1.set_xlim(1961,2021)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax1.set_xlabel('Year')
ax1.grid(ls='--')
ax1.axvspan(2002.5,2021,color='lightblue',alpha=0.3)
ax1.axvspan(1960,2002.5,color='#FFDE91',alpha=0.2)
ax1.legend(loc='upper left')
ax1.text(1996,0.16,'TCC={:.2f} (1985-2021)'.format(tcc_P_1),fontsize=10,horizontalalignment='left',fontweight='bold',color='r')
ax1.text(1996,0.13,'MSSS={:.2f} (1985-2021)'.format(msss_P_1),fontsize=10,horizontalalignment='left',fontweight='bold',color='r')
ax1.text(2004,-0.15,'TCC={:.2f} (2003-2021)'.format(tcc_P_2),fontsize=10,horizontalalignment='left',fontweight='bold',color='b')
ax1.text(2004,-0.18,'MSSS={:.2f} (2003-2021)'.format(msss_P_2),fontsize=10,horizontalalignment='left',fontweight='bold',color='b')
#---------------------------------------------(b)--------------------------------------------------
ax2=fig.add_axes([0.099,0.133,0.348,0.389])
ax2.set_title('b',loc='left',fontweight='bold')
ax2.plot(perfect_TCC,lw=lw,c='crimson',marker='.',label='Perfect prediction')
ax2.plot(hybrid_TCC,lw=lw,c='steelblue',marker='.',label='Hybrid prediction')
ax2.plot(persistent_TCC,lw=lw,c='k',marker='.',label='Persistent prediction')
ax2.plot(direct_TCC,lw=lw,c='seagreen',marker='.',label='Model direct prediction')
ax2.scatter(range(7),perfect_TCC,s=s,marker='*',c='crimson') #完美预测全部通过检验
ax2.scatter(range(5),hybrid_TCC[:5],s=s,marker='*',c='steelblue') #动力统计结合预测前5个通过检验
ax2.grid(ls='--')
ax2.set_xlim(0,6)
plt.xticks(range(7),['1-4','2-5','3-6','4-7','5-8','6-9','7-10'])
ax2.set_ylim(-1,1)
ax2.axhline(0,c='grey')
ax2.legend(loc='lower left',fontsize=7.5)
ax2.set_xlabel('Lead time (year)')
ax2.set_ylabel('Skill')
#---------------------------------------------(c)--------------------------------------------------
ax3=fig.add_axes([0.569,0.133,0.348,0.389])
ax3.set_title('c',loc='left',fontweight='bold')
ax3.plot(perfect_MSSS,lw=lw,c='crimson',marker='.',label='Perfect prediction')
ax3.plot(hybrid_MSSS,lw=lw,c='steelblue',marker='.',label='Hybrid prediction')
ax3.plot(persistent_MSSS,lw=lw,c='k',marker='.',label='Persistent prediction')
ax3.plot(direct_MSSS,lw=lw,c='seagreen',marker='.',label='Model direct prediction')
ax3.grid(ls='--')
ax3.set_xlim(0,6)
plt.xticks(range(7),['1-4','2-5','3-6','4-7','5-8','6-9','7-10'])
ax3.set_ylim(-1.5,0.75)
ax3.axhline(0,c='grey')
ax3.legend(loc='lower left',fontsize=7.5)
ax3.set_xlabel('Lead time (year)')
ax3.set_ylabel('Skill')

plt.savefig('./figure/Figure 4.pdf', bbox_inches='tight')