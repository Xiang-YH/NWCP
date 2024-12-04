import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import scipy.stats as st
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes
from mpl_toolkits.axisartist import grid_finder
import palettable
import geopandas as gpd
import shapely.ops as ops
from shapely.geometry import Polygon

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

def calc_PCC(obs,mod):
    '''
    :param obs: 观测结果
    :param mod: 模式结果
    :return: 模式的PCC技巧
    '''
    val=~np.isnan(mod)
    return st.pearsonr(obs[val],mod[val])[0]

def calc_NRMSE(obs,mod):
    '''
    :param obs: 观测结果
    :param mod: 模式结果
    :return: 模式的NRMSE技巧
    '''
    val=~np.isnan(mod)
    return np.sqrt(((obs[val]-mod[val])**2).mean())/np.std(obs[val])

def calc_RMSE(obs,mod):
    '''
    :param obs: 观测结果
    :param mod: 模式结果
    :return: 模式的RMSE技巧
    这里为了跟泰勒图相匹配，实际上算的是中心化RMSE（https://zhuanlan.zhihu.com/p/340501434）
    '''
    val=~np.isnan(mod)
    std_r=np.std(obs[val])
    Mod=mod/std_r #泰勒图的坐标是标准化的标准差，这里所有的结果也都出以观测标准差以进行标准化
    Obs=obs/std_r
    E_bar_2=(np.mean(Mod[val])-np.mean(Obs[val]))**2
    E_2=np.mean((Obs[val]-Mod[val])**2)
    return np.sqrt(E_2-E_bar_2)

models_res100km=np.array(['CESM2','CESM2-WACCM','EC-Earth3','EC-Earth3-CC','EC-Earth3-Veg-LR','EC-Earth3-Veg','FIO-ESM-2-0',
                       'GFDL-ESM4','MRI-ESM2-0']) #分辨率100km的模式
models_res250km=np.array(['ACCESS-CM2','ACCESS-ESM1-5','IPSL-CM6A-LR','KACE-1-0-G','MPI-ESM1-2-LR']) #分辨率250km的模式
models=np.concatenate((models_res100km,models_res250km), axis=0) #所有模式
models.sort()

#%%计算观测降水平均态和区域平均降水
fpre=xr.open_dataset('D:/data/xibei/CN05.1_Pre_1961_2021_monthly_1x1_extend.nc')
pre=fpre.pre.sel(time=slice('1961','2021')) #读取1961-2021年
pre=pre.sel(time=pre.time.dt.season=='JJA') #筛选JJA
lon=pre.lon.values
lat=pre.lat.values

mask_xb=np.load('./data/maskXB.npy')
pre.values=pre.values*mask_xb #保留西北地区
pre_JJA=pre.coarsen(time=3).mean()  #求逐年夏季平均
pre_JJA_mean=pre_JJA.mean('time').values  #夏季降水平均态
pre_JJA_a=pre_JJA-pre_JJA_mean #夏季降水异常

pre_hd=pre_JJA_a.rolling(time=4).mean() #进行4年滑动平均
pre_4=pre_hd[3:].values #4年滑动平均后前三个值是nan

pre_4_arealmean=arealmean(pre_4,lon,lat,[70,125],[25,60]) #年代际区域平均降水

#%%计算每个模式的降水平均态和区域平均序列
#先计算每个模式的成员平均SMEM，作为该模式的结果，再计算所有模式的平均MMEM
path="D:/data/xibei/CMIP6/high-res/"
files=os.listdir(path)

model_files = {} #用字典保存每个模式的所有文件，key是模式名，value是这个模式的文件名
for filename in files:
    model_name = filename.split("_")[4] #文件名中下划线分割的第5个元素是模式名
    print(model_name,end=',')
    if model_name not in model_files:
        model_files[model_name] = []
    model_files[model_name].append(filename)

MME_decadal = [] #所有模式的年代际区域平均降水序列集合
MME_mean = [] #所有模式的平均态集合
model_decadal={} #每个模式对应的年代际区域平均降水
model_mean={} #每个模式对应的降水平均态
for model_name, file_list in model_files.items():
    SME_decadal = [] #当前模式所有成员的年代际区域平均降水序列集合
    SME_mean=[]  #所有模式的平均态集合
    for file in file_list:
        fpr=xr.open_dataset(path+file)
        pr=fpr.pr.sel(time=slice('1961','2021'),lat=slice(14.5,55.5),lon=slice(70,140))
        pr=pr.sel(time=pr.time.dt.season=='JJA')  #筛选JJA
        pr.values=pr.values*mask_xb  #保留西北地区
        pr_JJA=pr.coarsen(time=3).mean()  #求逐年夏季平均
        pr_JJA_mean=pr_JJA.mean('time').values #夏季降水平均态
        pr_JJA_a=pr_JJA-pr_JJA_mean #降水异常
        pr_hd=pr_JJA_a.rolling(time=4).mean()  #降水异常进行4年滑动平均
        pr_4=pr_hd[3:].values  #4年滑动平均后前三个值是nan
        pr_decadal=arealmean(pr_4,lon,lat,[70,125],[25,60]) #滑动平均的区域平均序列
        SME_decadal.append(pr_decadal) #存入当前成员的滑动平均区域平均序列
        SME_mean.append(pr_JJA_mean)  #存入当前成员的降水量平均态

    SMEM_decadal=np.mean(SME_decadal,0) #当前模式所有成员的年代际区域平均降水序列集合平均
    model_decadal[model_name]=SMEM_decadal #保存当前模式的年代际区域平均降水
    MME_decadal.append(SMEM_decadal)
    SMEM_mean=np.mean(SME_mean,0) #当前模式所有成员的夏季降水量平均态集合平均
    model_mean[model_name]=SMEM_mean  #保存当前模式的夏季降水量平均态
    MME_mean.append(SMEM_mean)

MME_decadal=np.array(MME_decadal)
MMEM_decadal=np.mean(MME_decadal,0) #所有模式的年代际区域平均降水序列集合平均
MME_mean=np.array(MME_mean)
MMEM_mean=np.mean(MME_mean,0) #所有模式的夏季降水量平均态集合平均

#%%画每个模式的降水量平均态
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 13


def zitu(n,title,PCC,NRMSE,color,A):
    ax=plt.subplot(5,4,n,projection=ccrs.PlateCarree())
    ax.set_extent([70,122,27,55],crs=ccrs.PlateCarree())
    ax.set_title(title,loc='left',color=color)
    if PCC:
        ax.set_title('PCC={:.2f}\nRMSE={:.2f}'.format(PCC,NRMSE),fontsize=8,loc='right')
    cf=ax.contourf(lon,lat,A,lines,transform=ccrs.PlateCarree(),
                  cmap=palettable.scientific.diverging.Roma_20.mpl_colormap,extend='max')
    CNshp=gpd.read_file("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_country.shp")
    china_geometry=ops.unary_union(CNshp.geometry)
    world_extent=Polygon([(-180,-90),(180,-90),(180,90),(-180,90)])
    non_china_geometry=world_extent.difference(china_geometry)
    ax.add_geometries([non_china_geometry],ccrs.PlateCarree(),
                      facecolor='white',edgecolor='none',zorder=1)

    TPshp=Reader("./data/TP_mask.shp")
    ax.add_geometries(TPshp.geometries(),crs=ccrs.PlateCarree(),edgecolor='none',linewidths=0,facecolor='0.9')
    shp=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
    ax.add_geometries(shp.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=1,facecolor='none')
    NWC_shp=Reader("./data/NWC.shp")
    ax.add_geometries(NWC_shp.geometries(),crs=ccrs.PlateCarree(),edgecolor='darkred',linewidths=1.5,facecolor='none')
    return cf

fig=plt.figure(figsize=(12,9.5))
plt.subplots_adjust(left=0.05, bottom=0.07, right=0.95, top=0.95, wspace=0.1, hspace=0.2)
lines=np.arange(0,3.1,0.2)
c=zitu(1,'Observation',False,False,'k',pre_JJA_mean) #观测平均态

n=1
name_color=[] #画泰勒图的legend用
letter=['b','c']
for mod in models:
    n+=1
    while n in [2,3,6,7]: #给泰勒图空出位置
        n+=1

    PCC_mod=calc_PCC(pre_JJA_mean,model_mean[mod])
    RMSE_mod=calc_RMSE(pre_JJA_mean,model_mean[mod])
    if PCC_mod<0.7 or RMSE_mod>1.2:
        name_color.append('grey')
        zitu(n,mod,PCC_mod,RMSE_mod,'grey',model_mean[mod])
    else:
        name_color.append('k')
        zitu(n,mod,PCC_mod,RMSE_mod,'k',model_mean[mod])

cbar_ax=fig.add_axes([0.14,0.03,0.75,0.02])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)
cbar.set_ticks(lines)
cbar.ax.tick_params(labelsize=10)


#%%画泰勒图
def set_tayloraxes(fig,location):
    trans = PolarAxes.PolarTransform()
    r1_locs = np.hstack((np.arange(1,10)/10.0,[0.95,0.99]))
    t1_locs = np.arccos(r1_locs)
    gl1 = grid_finder.FixedLocator(t1_locs)
    tf1 = grid_finder.DictFormatter(dict(zip(t1_locs, map(str,r1_locs))))
    r2_locs=np.arange(0,3.01,0.5)
    r2_labels=['0','0.50','REF','1.50','2.00','2.50', '3.00']
    gl2 = grid_finder.FixedLocator(r2_locs)
    tf2 = grid_finder.DictFormatter(dict(zip(r2_locs, map(str,r2_labels))))
    ghelper = floating_axes.GridHelperCurveLinear(trans,extremes=(0,np.pi/2,0,3),
                                                  grid_locator1=gl1,tick_formatter1=tf1,
                                                  grid_locator2=gl2,tick_formatter2=tf2)
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=ghelper)
    ax.set_position(location)
    fig.add_subplot(ax)
    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")
    ax.axis["top"].major_ticklabels.set_fontsize(8)
    ax.axis["left"].set_axis_direction("bottom")
    ax.axis["left"].label.set_text("Standard deviation")
    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")
    ax.axis["bottom"].set_visible(False)
    ax.grid()
    polar_ax = ax.get_aux_axes(trans)
    t = np.linspace(0,np.pi/2)
    r = np.zeros_like(t) + 1
    polar_ax.plot(t,r,'k--')
    polar_ax.text(np.pi/2+0.042,1.03, "1.00", size=8,ha="right", va="top",
                  bbox=dict(boxstyle="square",ec='none',fc='w'))

    r=np.linspace(0,3,100)  # 半径从0到2
    theta=np.linspace(0,np.pi/2,100)  # 角度从0到2π
    R,Theta=np.meshgrid(r,theta)  # 创建网格
    Z=np.sqrt(R**2+1-2*R*np.cos(Theta)) #计算RMSE（https://zhuanlan.zhihu.com/p/340501434）

    contour = polar_ax.contour(Theta,R,Z,np.arange(0.5,3.01,0.5),colors='steelblue',
                               linestyles='dashed',linewidths=0.8) #画出RMSE的线
    polar_ax.clabel(contour,inline=True,fontsize=7, colors='steelblue', fmt='%.2f')
    polar_ax.contour(Theta,R,Z,[1.2],colors='green',linestyles='dashed',linewidths=2)

    polar_ax.plot([np.arccos(0.7)]*len(r),r,c='green',lw=2,ls='--')

    return polar_ax

def plot_taylor(axes, refsample, sample, *args, **kwargs):
    val=~np.isnan(refsample)
    std = np.std(sample[val])/np.std(refsample[val]) #模式结果都除以观测标准差，确保
    corr = np.corrcoef(refsample[val], sample[val])
    theta = np.arccos(corr[0,1])
    print(theta,std)
    t,r = theta,std
    d = axes.scatter(t,r, *args, **kwargs)
    return d


ax1 = set_tayloraxes(fig,[0.328,0.658,0.226,0.286])
ax1.set_title('')
colors = ['#172C51','#495C83','#7A86B6','#2F8F9D','#82DBD8','#90C8AC','#C4DFAA',
          '#A7DA18','#F9E0BB','#E7B10A','#FCC8D1','#D14D72','#D6481C','#DB005B',]
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
for mod,c,m in zip(models,colors,markers):
    d=plot_taylor(ax1,pre_JJA_mean,model_mean[mod],color=c,marker=m,label=mod,s=60,zorder=1)

legend=plt.legend(loc=(1.12,-0.05),frameon=False, labelspacing=0.8, fontsize=8)
texts=legend.get_texts()
for c,text in zip(name_color,texts):
    text.set_color(c)

plt.savefig('./figure/Supplementary Figure 1.pdf', bbox_inches='tight')
