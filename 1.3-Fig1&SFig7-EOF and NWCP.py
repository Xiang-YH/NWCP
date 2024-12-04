import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from matplotlib import ticker
from cartopy.io.shapereader import Reader
import os
from eofs.standard import Eof
import geopandas as gpd
import shapely.ops as ops
from shapely.geometry import Polygon

#%%
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


mask_xb=np.load('./data/maskXB.npy')
path="D:/data/xibei/CMIP6/high-res/"
files=os.listdir(path)

#剔除平均态PCC<0.7或NRMSE>1.2的模式
models=np.array(['EC-Earth3','EC-Earth3-CC','EC-Earth3-Veg-LR','EC-Earth3-Veg',
                       'GFDL-ESM4','MRI-ESM2-0','ACCESS-CM2','IPSL-CM6A-LR','KACE-1-0-G'])

#%%计算观测降水异常序列(P_bar_n)
fpre=xr.open_dataset('D:/data/xibei/CN05.1_Pre_1961_2021_monthly_1x1.nc')
pre=fpre.pre.sel(time=slice('1961','2021')) #读取1961-2021年
pre=pre.sel(time=pre.time.dt.season=='JJA') #筛选JJA
lon=pre.lon.values
lat=pre.lat.values

pre.values=pre.values*mask_xb #保留西北地区
pre_JJA=pre.coarsen(time=3).mean()  #求逐年夏季平均
pre_mean=pre_JJA.mean('time')
pre_JJA_a=pre_JJA-pre_mean

pre_hd=pre_JJA_a.rolling(time=4).mean() #进行4年滑动平均
pre_4=pre_hd[3:].values #4年滑动平均后前三个值是nan

OBS_decadal=arealmean(pre_4,lon,lat,[70,125],[25,60])

#%% 距平百分率的EOF分解
fpre=xr.open_dataset('D:/data/xibei/CN05.1_Pre_1961_2021_monthly_1x1_extend.nc')
pre=fpre.pre.sel(time=slice('1961','2021')) #读取1961-2021年
pre=pre.sel(time=pre.time.dt.season=='JJA') #筛选JJA
lon=pre.lon.values
lat=pre.lat.values

pre.values=pre.values*mask_xb #保留西北地区
pre_JJA=pre.coarsen(time=3).mean()  #求逐年夏季平均
pre_mean=pre_JJA.mean('time')
pre_JJA_a=pre_JJA-pre_mean
pre_jpbfl=pre_JJA_a/pre_mean  #距平百分率
pre_hd=pre_jpbfl.rolling(time=4).mean()  #进行4年滑动平均
pre_4=pre_hd[3:].values  #4年滑动平均后前三个值是nan

weights_array=np.cos(np.deg2rad(pre.lat.values))[:,np.newaxis]
solver=Eof(pre_4,weights=weights_array)
nmt=3
Z=solver.pcs(npcs=nmt)
S=solver.eofs(neofs=nmt)
r=solver.varianceFraction(nmt)
print("r=",r)
G=np.zeros(nmt)  #累积方差贡献率
for mt in range(nmt):
    G[mt]=np.sum(r[:mt+1])  #前mt个方差贡献率的和
print('G=',G)

PC1,EOF1=-Z[:,0],-S[0]

print(st.pearsonr(OBS_decadal,PC1))

#%%计算模式降水MME降水异常序列（Pm_bar_n）
#先计算每个模式的成员平均SMEM，作为该模式的结果，再计算所有模式的平均MMEM
model_files = {} #用字典保存每个模式的所有文件，key是模式名，value是这个模式的文件名
for filename in files:
    model_name = filename.split("_")[4] #文件名中下划线分割的第5个元素是模式名
    if model_name in models: #只读取模式列表中的模式数据
        if model_name not in model_files:
            model_files[model_name] = []
        model_files[model_name].append(filename)

MME_decadal = [] #所有模式的集合
for model_name, file_list in model_files.items():
    SME_decadal = [] #当前模式所有成员的集合
    # print()
    for file in file_list:
        # print(path+file)
        fpr=xr.open_dataset(path+file)
        pr=fpr.pr.sel(time=slice('1961','2021'),lat=slice(14.5,55.5),lon=slice(70,140))
        lon_1=pr.lon.values
        lat_1=pr.lat.values
        pr=pr.sel(time=pr.time.dt.season=='JJA')  #筛选JJA
        pr.values=pr.values*mask_xb  #保留西北地区
        pr_JJA=pr.coarsen(time=3).mean()  #求逐年夏季平均
        pr_JJA_a=pr_JJA-pr_JJA.mean('time') #降水异常
        pr_hd=pr_JJA_a.rolling(time=4).mean()  #降水异常进行4年滑动平均
        pr_4=pr_hd[3:].values  #4年滑动平均后前三个值是nan
        pr_decadal=arealmean(pr_4,lon_1,lat_1,[70,125],[25,60]) #滑动平均的区域平均序列
        SME_decadal.append(pr_decadal) #存入当前成员的滑动平均区域平均序列

    SMEM_decadal=np.mean(SME_decadal,0) #当前模式所有成员的集合平均
    # print(SMEM_decadal.shape)
    MME_decadal.append(SMEM_decadal)

MME_decadal=np.array(MME_decadal)
MMEM_decadal=np.mean(MME_decadal,0) #所有模式的集合平均


#%%计算bF值和内部变率（PI_bar_n）
bF_bar=st.linregress(MMEM_decadal,OBS_decadal)[0]
P_In=OBS_decadal-bF_bar*MMEM_decadal #计算内部变率分量，P_I=P-bF*Tm

#%%计算外强迫的置信区间
model_std=np.std(MME_decadal*bF_bar,0) #每一年的所有模式结果的std
Z_a2=1.96
N=len(model_files) #样本数=模式个数
d=model_std/np.sqrt(N)*Z_a2

#%%画图
fig=plt.figure(figsize=(12,7.5))
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10

#------------------------------------------------(a)----------------------------------------------
ax1=fig.add_axes([0.055,0.597,0.371,0.347],projection=ccrs.PlateCarree())
ax1.set_extent([70,125,25,55],crs=ccrs.PlateCarree())
ax1.set_title('a',loc='left',fontweight='bold',fontsize=15)
ax1.set_title('{:.2f}%'.format(r[0]*100),loc='right')

#画EOF1分布
lines=[-0.05,-0.03,0,0.03,0.05,0.07,0.09,0.13,0.18,0.24]
cmap=plt.get_cmap('BrBG')
color = [cmap(i) for i in (75,88,102,155,169,187,197,208,224,239,255)]
c=ax1.contourf(lon,lat,EOF1*np.std(PC1),lines,transform=ccrs.PlateCarree(),colors=color,extend='both')

#画中国和西北地区边界，只保留中国区域内的部分
CNshp=gpd.read_file("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
china_geometry=ops.unary_union(CNshp.geometry)
world_extent=Polygon([(-180,-90),(180,-90),(180,90),(-180,90)])
non_china_geometry=world_extent.difference(china_geometry)
ax1.add_geometries([non_china_geometry],ccrs.PlateCarree(),
                  facecolor='white',edgecolor='none',zorder=1)

TPshp=Reader("./data/TP_mask.shp")
ax1.add_geometries(TPshp.geometries(),crs=ccrs.PlateCarree(),edgecolor='none',linewidths=0,facecolor='0.9')
shp=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
ax1.add_geometries(shp.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=1,facecolor='none')
NWCshp=Reader("./data/NWC.shp")
ax1.add_geometries(NWCshp.geometries(),crs=ccrs.PlateCarree(),edgecolor='darkred',linewidths=1.2,facecolor='none')


#画坐标轴刻度
ax1.set_xticks(np.arange(75,130,10))  #指定要显示的经纬度
ax1.set_yticks(np.arange(25,60,10))
ax1.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(5))

#画colorbar
cbar_ax=fig.add_axes([0.065,0.555,0.350,0.015])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)
cbar.set_ticks(lines)
cbar.ax.tick_params(labelsize=9)

# ------------------------------------------------(b)----------------------------------------------
year=np.arange(1962,2020)
ax2=fig.add_axes([0.483,0.587,0.500,0.347])
ax2.set_title('b',loc='left',fontweight='bold',fontsize=15)
ax2.plot(year,PC1/np.std(PC1),label='PC1',c='seagreen',lw=2.5)
ax2.plot(year,OBS_decadal/np.std(OBS_decadal),label='NWCP',c='k',lw=2.5)
ax2.set_xticks(np.arange(1960,2021,10))
ax2.set_xlim(1961,2021)
ax2.set_ylabel('Intensity (standardized)')
ax2.set_xlabel('Year')
ax2.grid(ls='--')
ax2.axhline(0,linestyle='--',c='k',lw=1.2)
ax2.set_ylim(-3,3)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax2.legend()
ax2.text(1990,-2.4,'Corr(PC1, NWCP)={:.2f}'.format(np.corrcoef(OBS_decadal,PC1)[0,1]),fontsize=12,horizontalalignment='center',
         color='seagreen')


plt.savefig('./figure/Supplementary Figure 7.pdf', bbox_inches='tight')


#%%画图
fig=plt.figure(figsize=(12,7.5))
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10

ax3=fig.add_axes([0.175,0.079,0.682,0.393])
ax3.plot(np.arange(1962,2020),OBS_decadal,c='k',lw=4.5,label='NWCP',alpha=0.8)
ax3.plot(np.arange(1962,2020),MMEM_decadal*bF_bar,c='r',lw=4.5,label='NWCP-F',alpha=0.8)
plt.fill_between(np.arange(1962,2020),MMEM_decadal*bF_bar-d,MMEM_decadal*bF_bar+d,facecolor="orange",alpha=0.5,label='confidence interval')
ax3.plot(np.arange(1962,2020),P_In,c='royalblue',lw=4.5,label='NWCP-I',alpha=0.8)
ax3.axhline(0,ls='--',c='grey')
ax3.set_xlim(1961,2021)
ax3.set_xlabel('Year')
ax3.set_ylim(-0.2,0.2)
ax3.set_ylabel('Intensity (mm/day)')
ax3.grid(ls='--')
ax3.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

plt.legend(loc='upper left')

ax3.text(1980,-0.11,'Corr(NWCP-F, NWCP)={:.2f}'.format(np.corrcoef(OBS_decadal,MMEM_decadal*bF_bar)[0,1]),fontsize=12,horizontalalignment='center',
         color='r',ha='left')
ax3.text(1980,-0.15,'Corr(NWCP-I , NWCP)={:.2f}'.format(np.corrcoef(OBS_decadal,P_In)[0,1]),fontsize=12,horizontalalignment='center',
         color='royalblue',ha='left')

plt.savefig('./figure/Figure 1.pdf', bbox_inches='tight')

