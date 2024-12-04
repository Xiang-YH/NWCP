import numpy as np
import xarray as xr
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def weighted_average(sst,R,R90,lon,lat,pos):
    lon1,lon2,lat1,lat2=pos
    r=R.copy()
    r[np.isnan(r)]=0 #原本陆地区域r为缺测，现权重赋值为0
    B=sst.copy()
    B[np.isnan(B)]=0 #消除掉相关系数场里的nan，不然np.average没法算
    B=B[:,(lat>=lat1)&(lat<=lat2),:]
    B=B[:,:,(lon>=lon1)&(lon<=lon2)]
    r=r[(lat>=lat1)&(lat<=lat2),:]
    r=r[:,(lon>=lon1)&(lon<=lon2)]
    r[np.abs(r)<R90]=0  #未通过显著性检验的格点权重为0
    T=B.shape[0]
    C=np.ones(T)*np.nan
    weight=r.copy()
    weight[r!=0]=1  #只有权重不为0的格点参与计算
    for t in range(T):
        r_copy=r.copy()
        r_copy[r_copy>0]=1
        r_copy[r_copy<0]=-1 #不以相关系数为权重
        C[t]=np.average(B[t,:,:]*r_copy,weights=weight)
    return C

def bzh(x):
    return (x-np.mean(x))/np.std(x)


def calc_MSSS(OBS,Simu):
    MSE=np.mean((Simu-OBS)**2)
    MSEc=np.mean((OBS-np.mean(OBS))**2)
    return 1-MSE/MSEc

#%%读取观测数据
fsst=xr.open_dataset('D:/data/xibei/sst.mnmean.nc')
sst=fsst.sst.sel(time=slice('1967','2019'))
sst=sst.sel(time=sst.time.dt.season=='JJA')
sst_JJA=sst.coarsen(time=3).mean()
sst_hd=sst_JJA.rolling(time=4).mean() #进行4年滑动平均
sst_4=sst_hd[3:].values #4年滑动平均后,前三个值是nan

lon,lat=sst.lon,sst.lat
year=np.arange(1968,2017+1)

Pos=[
    [308,357,17,68],
    [123,283,-33,60],
    [34,120,-41,25],
] #预测因子的位置范围

Ps=['NAD','PDO','IO'] #预测因子的名字

R=np.load('./data/R_train.npy')

#%% 蒙特卡洛检验
R_95=0.4999
R_90=0.4305

#%%读取模式数据
fmod=xr.open_dataset('./data/tos_lead_moving_average.nc')
tos_1_4_decadal=fmod['tos_1-4_decadal'].values
tos_2_5_decadal=fmod['tos_2-5_decadal'].values
tos_3_6_decadal=fmod['tos_3-6_decadal'].values
tos_4_7_decadal=fmod['tos_4-7_decadal'].values
tos_5_8_decadal=fmod['tos_5-8_decadal'].values
tos_6_9_decadal=fmod['tos_6-9_decadal'].values
tos_7_10_decadal=fmod['tos_7-10_decadal'].values

tos_model=[tos_1_4_decadal,tos_2_5_decadal,tos_3_6_decadal,tos_4_7_decadal,tos_5_8_decadal,
           tos_6_9_decadal,tos_7_10_decadal]

leads=['1-4','2-5','3-6','4-7','5-8','6-9','7-10']

#%%计算模式预测因子并画折线图
indecator_TCC={} #key是预测因子名，value是该因子在不同提前期的值
indecator_MSSS={} #key是预测因子名，value是该因子在不同提前期的值

NAD_lead={}
for s,pos in zip(Ps,Pos):
    if s=='NAD':
        NAD_OBS=bzh(weighted_average(sst_4,R,R_90,lon,lat,pos))
    OBS=bzh(weighted_average(sst_4,R,R_90,lon,lat,pos))
    indecator_TCC[s]=[]
    indecator_MSSS[s]=[]

    for lead,tos_mod in zip(leads,tos_model):
        if s=='NAD':
            NAD_lead[lead]=bzh(weighted_average(tos_mod,R,R_90,lon,lat,pos))
        X=bzh(weighted_average(tos_mod,R,R_90,lon,lat,pos))
        TCC=st.pearsonr(OBS,X)[0]
        MSSS=calc_MSSS(OBS,X)
        indecator_TCC[s].append(TCC)
        indecator_MSSS[s].append(MSSS)

#%%画图
fig=plt.figure(figsize=(9,7.5))
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 12
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

#---------------------------------------------(a)--------------------------------------------------
ax1=fig.add_axes([0.099,0.613,0.818,0.319])
years=np.arange(1968,2017+1)
colors = ['#495C83','#2F8F9D','#90C8AC','#A7DA18','#E7B10A','#D14D72','#DB005B']

ax1.plot(years,NAD_OBS,lw=2,color='k',alpha=0.8,label='Observation')
for lead,color in zip(leads,colors):
    ax1.plot(years,NAD_lead[lead],lw=2,color=color,alpha=0.8,label=f'lead {lead} yr')
ax1.set_title('a',loc='left',fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylim(-2.5,2.5)
ax1.set_ylabel('Intensity (mm/day)')
ax1.grid(ls='--')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
ax1.legend(loc='lower left',ncol=2)
#---------------------------------------------(b)--------------------------------------------------
ax2=fig.add_axes([0.099,0.133,0.348,0.389])
C=['b','C2','C9','C4','C5','tomato','C3','C7']
ax2.set_title('b',loc='left',fontweight='bold')
ax2.plot(indecator_TCC['NAD'],marker='.')
ax2.grid(ls='--')
ax2.set_xlim(0,6)
plt.xticks(range(7),['1-4','2-5','3-6','4-7','5-8','6-9','7-10'])
ax2.set_ylim(-1,1)
ax2.axhline(0,c='grey')
ax2.set_xlabel('Lead time (year)')
ax2.set_ylabel('Skill')
#---------------------------------------------(c)--------------------------------------------------
ax3=fig.add_axes([0.569,0.133,0.348,0.389])
ax3.set_title('c',loc='left',fontweight='bold')
ax3.plot(indecator_MSSS['NAD'],marker='.')
ax3.grid(ls='--')
ax3.set_xlim(0,6)
plt.xticks(range(7),['1-4','2-5','3-6','4-7','5-8','6-9','7-10'])
ax3.set_ylim(-1,1)
ax3.axhline(0,c='grey')
ax3.set_xlabel('Lead time (year)')
ax3.set_ylabel('Skill')

plt.savefig('./figure/Supplementary Figure 5.pdf', bbox_inches='tight')

