import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import palettable
from metpy.units import units
import os


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
pre_JJA_a=pre_JJA-pre_JJA.mean('time')

pre_hd=pre_JJA_a.rolling(time=4).mean() #进行4年滑动平均
pre_4=pre_hd[3:].values #4年滑动平均后前三个值是nan

OBS_decadal=arealmean(pre_4,lon,lat,[70,125],[25,60])

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
#置信区间是MMEM_decadal*bF_bar-d到MMEM_decadal*bF_bar+d
#https://zhuanlan.zhihu.com/p/259232881  置信区间的计算


#%% 读取数据
def standardize(x):
    return (x-np.mean(x))/np.std(x)


years=np.arange(1962,2020)

fuvz=xr.open_dataset('D:/data/xibei/ERA5-UVZ-1961_2021-25_25.nc')
fpre=xr.open_dataset('D:/data/xibei/precip.mon.anom.nc')
fq=xr.open_dataset('D:/data/xibei/ERA5-q-1961_2021-25_25.nc')
fsst=xr.open_dataset('D:/data/xibei/sst.mnmean.nc')
sst=fsst.sst.sel(time=slice('1961','2021'),lat=slice(90,-90)) #用来读取经纬度

R_train=np.load('./data/R_train.npy') #训练时段内的降水内部变率与SST的相关系数场

def load_and_process(file, var_name, level=None,**kwargs):
    var = file[var_name].sel(time=slice('1961','2021'), **kwargs)
    if level is not None:
        var = var.sel(level=level)
    var_season = var.sel(time=var.time.dt.season == 'JJA')
    return var_season

z500=load_and_process(fuvz,'z',500,latitude=slice(90,-30))/9.8 #从位势变成位势高度（单位位势米）
z200=load_and_process(fuvz,'z',200,latitude=slice(90,-30))/9.8
u850=load_and_process(fuvz,'u',850,latitude=slice(90,-30))
v850=load_and_process(fuvz,'v',850,latitude=slice(90,-30))
u700=load_and_process(fuvz,'u',700,latitude=slice(90,-30))
v700=load_and_process(fuvz,'v',700,latitude=slice(90,-30))
u200=load_and_process(fuvz,'u',200,latitude=slice(90,-30))
v200=load_and_process(fuvz,'v',200,latitude=slice(90,-30))
pre=load_and_process(fpre,'precip',lat=slice(75,-30))

u=load_and_process(fuvz,'u',level=slice(300,700),latitude=slice(80,20),longitude=slice(50,150))*units('m/s')
v=load_and_process(fuvz,'v',level=slice(300,700),latitude=slice(80,20),longitude=slice(50,150))*units('m/s')
q=load_and_process(fq,'q',level=slice(300,700),latitude=slice(80,20),longitude=slice(50,150))*units('kg/kg')

u200_mean=u200.values.mean(0)
v200_mean=v200.values.mean(0)

u_850_700=(u850+u700)/2
v_850_700=(v850+v700)/2

#%%计算700hPa-300hPa水汽通量
from metpy.constants import g

qu=1/g*q*u
qv=1/g*q*v

WVFu=qu.integrate('level')*units('hPa')
WVFv=qv.integrate('level')*units('hPa')
WVFu.data=WVFu.data.to('kg/m/s')
WVFv.data=WVFv.data.to('kg/m/s')

#%% 求逐年夏季平均和滑动平均
def seasonal_mean_and_smooth(data):
    seasonal_mean = data.coarsen(time=3).mean() #对每年的三个月求平均
    smooth = seasonal_mean.rolling(time=4).mean() #求4年滑动平均
    return smooth[3:].values

z500_4=seasonal_mean_and_smooth(z500)
u200_4=seasonal_mean_and_smooth(u200)
v200_4=seasonal_mean_and_smooth(v200)
z200_4=seasonal_mean_and_smooth(z200)
u850_4=seasonal_mean_and_smooth(u850)
v850_4=seasonal_mean_and_smooth(v850)
u700_4=seasonal_mean_and_smooth(u700)
v700_4=seasonal_mean_and_smooth(v700)
pre_4=seasonal_mean_and_smooth(pre)
WVFu_4=seasonal_mean_and_smooth(WVFu)
WVFv_4=seasonal_mean_and_smooth(WVFv)

u_850_700_4=seasonal_mean_and_smooth(u_850_700)
v_850_700_4=seasonal_mean_and_smooth(v_850_700)


#%%提取训练时段
#训练时段1961-2002年，对应年代际序列的范围时1962-2000年
years_train=years[:-19]
print(years_train)

P_In_standardized=(P_In-np.mean(P_In))/np.std(P_In)
P_train=P_In_standardized[:-19]
z200_4_train=z200_4[:-19]
z500_4_train=z500_4[:-19]
v850_4_train=v850_4[:-19]
u850_4_train=u850_4[:-19]
v700_4_train=v700_4[:-19]
u700_4_train=u700_4[:-19]
pre_4_train=pre_4[:-19]
WVFu_4_train=WVFu_4[:-19]
WVFv_4_train=WVFv_4[:-19]

v_850_700_4_train=v_850_700_4[:-19]
u_850_700_4_train=u_850_700_4[:-19]

#%% 求回归
def regression(x,Y):
    T,I,J=Y.shape
    K=np.zeros([I,J])*np.nan
    R=np.zeros([I,J])*np.nan
    for i in range(I):
        for j in range(J):
            if ~np.isnan(Y[:,i,j]).any():
                K[i,j],_,R[i,j]=st.linregress(x,Y[:,i,j])[:3]
    return K,R


k_z200,r_z200=regression(P_train,z200_4_train)
k_z500,r_z500=regression(P_train,z500_4_train)
k_u850,r_u850=regression(P_train,u850_4_train)
k_v850,r_v850=regression(P_train,v850_4_train)
k_u700,r_u700=regression(P_train,u700_4_train)
k_v700,r_v700=regression(P_train,v700_4_train)
k_pre,r_pre=regression(P_train,pre_4_train)
k_WVFu,r_WVFu=regression(P_train,WVFu_4_train)
k_WVFv,r_WVFv=regression(P_train,WVFv_4_train)

k_u_850_700,r_u_850_700=regression(P_train,u_850_700_4_train)
k_v_850_700,r_v_850_700=regression(P_train,v_850_700_4_train)

#%% 蒙特卡洛检验
R_95=0.4999
R_90=0.4305

#%%计算波活动通量
def TN(za,u_mean,v_mean,lon,lat,p0):
    """
    计算波活动通量的x和y分量。

    :param za: 参数“za”代表位势高度异常。它是对给定位置和时间的位势高度与其平均值的偏差的度量。
    :param u_mean: 平均纬向风速，单位为米/秒。
    :param v_mean: 平均经向风速，单位为米/秒。
    :param lon: 网格点的经度值（以度为单位）。
    :param lat: 网格点的纬度值（以度为单位）。
    :param p0: p0 是表面压力，单位为 hPa（百帕斯卡）。
    :return: 使用给定参数计算的波活动通量的x和y分量。
    """
    a = 6400000 #地球半径
    omega = 7.292e-5 #自转角速度
    dlon = (np.gradient(lon)*np.pi/180.0).reshape((1,-1))
    dlat = (np.gradient(lat)*np.pi/180.0).reshape((-1,1))
    f = np.array(2*omega*np.sin(lat*np.pi/180.0)).reshape((-1,1)) #Coriolis parameter: f=2*omgega*sin(lat )
    cos_lat = (np.cos(np.array(lat)*np.pi/180)).reshape((-1,1)) #cos(lat)
    u_c=u_mean
    v_c=v_mean
    g=9.8
    psi_p = g*za/f #Pertubation stream-function
    #5 partial differential terms
    dpsi_dlon = np.gradient(psi_p, axis=1)/dlon
    dpsi_dlat = np.gradient(psi_p, axis=0)/dlat
    d2psi_dlon2=np.gradient(dpsi_dlon, axis=1)/dlon
    d2psi_dlat2=np.gradient(dpsi_dlat, axis=0)/dlat
    d2psi_dlondlat = np.gradient(dpsi_dlon, axis=0)/dlat
    termxu=dpsi_dlon*dpsi_dlon-psi_p*d2psi_dlon2
    termxv=dpsi_dlon*dpsi_dlat-psi_p*d2psi_dlondlat
    termyv=dpsi_dlat*dpsi_dlat-psi_p*d2psi_dlat2
    #coefficient
    p= p0/1000
    magU=np.sqrt(u_c**2+v_c**2)
    coeff=p*cos_lat/(2*a*a*magU)
    #x-component of TN-WAF
    px = coeff * ((u_c/cos_lat/cos_lat)*termxu+v_c*termxv/cos_lat)
    #y- component of TN-WAF
    py = coeff * ((u_c/cos_lat)*termxv+v_c*termyv)
    return px,py

TNx,TNy=TN(k_z200,u200_mean,v200_mean,z200.longitude,z200.latitude,200)
TNx[22:,:]=np.nan
TNy[22:,:]=np.nan

#%%画图
fig=plt.figure(figsize=(12,8.5))
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 15

#---------------------------------------------(a)------------------------------------------
lonz,latz=z200.longitude,z200.latitude
k_z200_cyclic,lonz_cyclic=add_cyclic_point(k_z200, coord=lonz)
r_z200_cyclic,_=add_cyclic_point(r_z200, coord=lonz)
Lonz_cyclic,Latz=np.meshgrid(lonz_cyclic,latz)
lines=np.arange(-9,9.1,1.5)

ax1=fig.add_axes([0.050,0.790,0.590,0.170],projection=ccrs.PlateCarree(110))
ax1.set_aspect('auto') #把固定长宽比改为自由长宽比
ax1.set_title('a',loc="left",fontweight='bold')
ax1.set_extent([0,359,20,80], crs=ccrs.PlateCarree())
ax1.coastlines('110m', alpha=0.8,color='0.2')

cmap=plt.cm.get_cmap('PuOr_r')
color=[cmap(i) for i in (10,15,25,35,55,75,100,256-125,256-95,256-70,256-63,256-50,256-35,256-20)]
c=ax1.contourf(lonz_cyclic,latz,k_z200_cyclic,lines,
               transform=ccrs.PlateCarree(),colors=color,extend='both',alpha=0.7)
ax1.scatter(Lonz_cyclic[np.abs(r_z200_cyclic)>R_90],Latz[np.abs(r_z200_cyclic)>R_90],marker='o', ############ R90
            color='w',s=2,alpha=0.4,transform=ccrs.PlateCarree())
arrow_lengths = np.sqrt(TNx**2 + TNy**2)
TNx[arrow_lengths<0.01]=np.nan
quiv1=ax1.quiver(lonz[::2],latz[::2],TNx[::2,::2],TNy[::2,::2],transform=ccrs.PlateCarree()
             ,headwidth=4,headlength=6,scale=2,width=0.0018,color='darkgreen',alpha=0.9)
ax1.quiverkey(quiv1,0.95,1.08,0.05,"0.05",fontproperties={'size':11},labelpos='W')

shp_CN=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
ax1.add_geometries(shp_CN.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=0.8,facecolor='none')
shp_NWC=Reader("./data/NWC.shp")
ax1.add_geometries(shp_NWC.geometries(),crs=ccrs.PlateCarree(),edgecolor='darkred',linewidths=1.3,facecolor='none')

ax1.set_xticks(np.arange(-180,181,30))#指定要显示的经纬度
ax1.set_yticks(np.arange(20,81,15))
ax1.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(5))

cbar_ax=fig.add_axes([0.120,0.750,0.450,0.010])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)
cbar.set_ticks(lines)
cbar.ax.tick_params(labelsize=7)
#
# #---------------------------------------------(b)------------------------------------------
k_z500_cyclic,lonz_cyclic=add_cyclic_point(k_z500, coord=lonz)
r_z500_cyclic,_=add_cyclic_point(r_z500, coord=lonz)
lines=np.arange(-6,6.1,1)

ax2=fig.add_axes([0.050,0.520,0.590,0.170],projection=ccrs.PlateCarree(110))
ax2.set_aspect('auto') #把固定长宽比改为自由长宽比
ax2.set_title('b',loc="left",fontweight='bold')
ax2.set_extent([0,359,20,80], crs=ccrs.PlateCarree())
ax2.coastlines('110m', alpha=0.8,color='0.2')
#
cmap=plt.cm.get_cmap('PuOr_r')
color=[cmap(i) for i in (5,10,20,30,50,70,95,256-120,256-90,256-65,256-58,256-45,256-30,256-15)]
c=ax2.contourf(lonz_cyclic,latz,k_z500_cyclic,lines,
               transform=ccrs.PlateCarree(),colors=color,extend='both',alpha=0.7)
ax2.scatter(Lonz_cyclic[np.abs(r_z500_cyclic)>R_90],Latz[np.abs(r_z500_cyclic)>R_90],marker='o', ############ R90
            color='w',s=2,alpha=0.4,transform=ccrs.PlateCarree())
arrow_lengths = np.sqrt(k_u700**2 + k_v700**2)
k_u700[arrow_lengths<0.1]=np.nan
quiv2=ax2.quiver(lonz[::2],latz[::2],k_u700[::2,::2],k_v700[::2,::2],transform=ccrs.PlateCarree()
             ,headwidth=4,headlength=6,scale=20,width=0.0014,color='k',alpha=0.7)
ax2.quiverkey(quiv2,0.95,1.08,0.5,"0.5",fontproperties={'size':11},labelpos='W')

shp_CN=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
ax2.add_geometries(shp_CN.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=0.8,facecolor='none')
shp_NWC=Reader("./data/NWC.shp")
ax2.add_geometries(shp_NWC.geometries(),crs=ccrs.PlateCarree(),edgecolor='darkred',linewidths=1.3,facecolor='none')
ax2.set_xticks(np.arange(-180,181,30))#指定要显示的经纬度
ax2.set_yticks(np.arange(20,81,15))
ax2.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式
ax2.yaxis.set_major_formatter(LatitudeFormatter())
ax2.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(5))

cbar_ax=fig.add_axes([0.120,0.480,0.450,0.010])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)
cbar.set_ticks(lines)
cbar.ax.set_xticklabels([f'{x:.1f}' for x in lines]) #显示时保留一位小数
cbar.ax.tick_params(labelsize=7)
#
# #---------------------------------------------(c)------------------------------------------
lonpre,latpre=pre.lon,pre.lat
lonq,latq=q.longitude,q.latitude
Lonpre,Latpre=np.meshgrid(lonpre,latpre)

lines=[-0.25,-0.2,-0.15,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.15,0.2,0.25]

ax3=fig.add_axes([0.050,0.190,0.180,0.230],projection=ccrs.PlateCarree(180))
ax3.set_aspect('auto') #把固定长宽比改为自由长宽比
ax3.set_title('c',loc="left",fontweight='bold')
ax3.set_extent([70,137,17,65], crs=ccrs.PlateCarree())
ax3.coastlines('110m', alpha=0.8,color='0.2')

cmap=plt.cm.get_cmap('BrBG')
color = [cmap(i) for i in (40,60,75,90,105,120,256-110,256-95,256-75,256-55,256-45,256-35)]
color.insert(6,(1,1,1,1)) #中间颜色改成白色

c=ax3.contourf(lonpre,latpre,k_pre,lines,
               transform=ccrs.PlateCarree(),colors=color,extend='both',alpha=0.85)
ax3.scatter(Lonpre[np.abs(r_pre)>R_90],Latpre[np.abs(r_pre)>R_90],marker='o', ############ R90
            color='w',s=1.5,alpha=0.8,transform=ccrs.PlateCarree())
arrow_lengths = np.sqrt(k_u_850_700**2 + k_v_850_700**2)
k_u_850_700[arrow_lengths<0.03]=np.nan
quiv3=ax3.quiver(lonz,latz,k_u_850_700,k_v_850_700,transform=ccrs.PlateCarree()
             ,headwidth=4,headlength=6,scale=3.5,width=0.004,color='k')
ax3.quiverkey(quiv3,0.92,1.03,0.3,"0.3",fontproperties={'size':9},labelpos='W')

shp_CN=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
ax3.add_geometries(shp_CN.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=1,facecolor='none')
shp_NWC=Reader("./data/NWC.shp")
ax3.add_geometries(shp_NWC.geometries(),crs=ccrs.PlateCarree(),edgecolor='darkred',linewidths=1.5,facecolor='none')

ax3.set_xticks(np.arange(-110,-40,15))#指定要显示的经纬度
ax3.set_yticks(np.arange(20,61,10))
ax3.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式
ax3.yaxis.set_major_formatter(LatitudeFormatter())
ax3.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax3.yaxis.set_minor_locator(mticker.MultipleLocator(5))


cbar_ax=fig.add_axes([0.050,0.150,0.180,0.010])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)
cbar.ax.tick_params(labelsize=7)

#---------------------------------------------(d)------------------------------------------
lines=[-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
lon,lat=sst.lon,sst.lat
lonsst,latsst=np.meshgrid(lon,lat)
R_cyclic,lon_cyclic=add_cyclic_point(R_train, coord=lon)

ax4=fig.add_axes([0.270,0.190,0.370,0.230],projection=ccrs.PlateCarree(110))
ax4.set_aspect('auto') #把固定长宽比改为自由长宽比
ax4.set_extent([0,359,-50,75], crs=ccrs.PlateCarree())
ax4.coastlines('110m', alpha=0.8,color='0.2',zorder=2)
ax4.add_feature(cfeature.LAND,color='0.9',zorder=1)
ax4.set_title('d',loc='left',fontweight='bold')

cmap=palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap
color = [cmap(i) for i in (10,20,30,45,65,80,95,256-110,256-105,256-95,256-80,256-60,256-35,256-30)]
color.insert(7,(1,1,1,1)) #中间颜色改成白色

c=ax4.contourf(lon_cyclic,lat,R_cyclic,lines,
               transform=ccrs.PlateCarree(),colors=color,extend='both',alpha=0.7,zorder=2)
ax4.scatter(lonsst[np.abs(R_train)>R_90],latsst[np.abs(R_train)>R_90],marker='o', ############ R90
            color='0.1',s=0.7,alpha=0.4,transform=ccrs.PlateCarree(),zorder=2)



def draw_location(ax,color,lon1,lon2,lat1,lat2,lw,alpha):
    ax.plot([lon1,lon1],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree(),alpha=alpha)
    ax.plot([lon2,lon2],[lat1,lat2],lw=lw,c=color,transform=ccrs.PlateCarree(),alpha=alpha)
    ax.plot([lon1,lon2],[lat1,lat1],lw=lw,c=color,transform=ccrs.PlateCarree(),alpha=alpha)
    ax.plot([lon1,lon2],[lat2,lat2],lw=lw,c=color,transform=ccrs.PlateCarree(),alpha=alpha)

#NAD
lon1,lon2,lat1,lat2=308,357,17,66
draw_location(ax4,'C3',lon1,lon2,lat1,lat2,2.3,1)
ax4.text((lon1+lon2)/2,(lat1+lat2)/2,'NAD',c='C3',fontsize=12,transform=ccrs.PlateCarree(),ha='center',va='center')

#PDO
lon1,lon2,lat1,lat2=123,283,-33,60
draw_location(ax4,'C2',lon1,lon2,lat1,lat2,1.5,0.8)
ax4.text((lon1+lon2)/2,(lat1+lat2)/2,'PDO',c='C2',fontsize=12,transform=ccrs.PlateCarree(),ha='center',va='center')

#IO
lon1,lon2,lat1,lat2=34,120,-41,25
draw_location(ax4,'b',lon1,lon2,lat1,lat2,1.5,0.8)
ax4.text((lon1+lon2)/2,(lat1+lat2)/2,'IO',c='b',fontsize=12,transform=ccrs.PlateCarree(),ha='center',va='center')


#设置坐标轴刻度
ax4.set_xticks(np.arange(-180,181,40))#指定要显示的经纬度
ax4.set_yticks(np.arange(-45,80,15))
ax4.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式
ax4.yaxis.set_major_formatter(LatitudeFormatter())
ax4.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax4.yaxis.set_minor_locator(mticker.MultipleLocator(5))


cbar_ax=fig.add_axes([0.270,0.150,0.370,0.010])
cbar=plt.colorbar(c,orientation='horizontal',cax=cbar_ax)
cbar.set_ticks(lines)
cbar.ax.tick_params(labelsize=7)

plt.savefig('./figure/Figure 2.pdf', bbox_inches='tight')
