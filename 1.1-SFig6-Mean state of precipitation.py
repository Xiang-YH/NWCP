import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
import matplotlib.ticker as mticker
import palettable
import geopandas as gpd
from matplotlib.path import Path
from cartopy.mpl.patch import geos_to_path
from shapely.geometry import LineString, Polygon, Point


#%%计算降水量平均态
fpre=xr.open_dataset('D:/data/xibei/CN05.1_Pre_1961_2021_monthly_1x1_extend.nc') #extend数据把非nan格点向外扩充了一个点，画图好看
pre=fpre.pre.sel(time=slice('1961','2021')) #读取1961-2021年
pre=pre.sel(time=pre.time.dt.season=='JJA') #筛选JJA
pre_mean=pre.mean('time')
lon=pre.lon.values
lat=pre.lat.values
Lon,Lat=np.meshgrid(lon,lat)


#%%画降水量平均态图
cmap=palettable.scientific.diverging.Roma_20.mpl_colormap
lines=np.arange(0,7.1,0.5)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False  #确保正确显示汉字和正负号
fig=plt.figure(figsize=(7,6))
ax1=plt.subplot(111,projection=ccrs.PlateCarree())
ax1.set_extent([72,137,17,55],crs=ccrs.PlateCarree())
ax1.set_title('1961-2021年夏季降水量平均态',fontsize=15)
ax1.add_feature(cfeature.OCEAN,color='lightblue')
cf=ax1.contourf(lon,lat,pre_mean,lines,transform=ccrs.PlateCarree(),cmap=cmap,extend='max')
c1=ax1.contour(lon,lat,pre_mean,[3],transform=ccrs.PlateCarree(),colors='r',linewidths=0.5)
pre_copy=pre_mean.copy()
I,J=pre_mean.shape
for i in range(I):
    for j in range(J):
        if Lon[i,j]<103 and Lat[i,j]>36 and pre_copy[i,j]>=3:
            pre_copy[i,j]=2.99999
        elif Lon[i,j]>88 and Lat[i,j]<30.1 and pre_copy[i,j]<=3:
            pre_copy[i,j]=3.00001
        elif Lon[i,j]>118.1 and Lat[i,j]<43.8 and pre_copy[i,j]<=3:
            pre_copy[i,j]=3.00001
        elif Lon[i,j]>119.5 and Lat[i,j]<45.5 and pre_copy[i,j]<=3:
            pre_copy[i,j]=3.00001
        elif Lon[i,j]>111 and Lat[i,j]<38 and pre_copy[i,j]<=3:
            pre_copy[i,j]=3.00001
        elif Lon[i,j]>106.8 and Lat[i,j]<35.9 and pre_copy[i,j]<=3:
            pre_copy[i,j]=3.00001
c2=ax1.contour(lon,lat,pre_copy,[3],transform=ccrs.PlateCarree(),colors='r',linewidths=1.5)

#只保留中国区域内的部分
CNshp=gpd.read_file("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
path_clip = Path.make_compound_path(*geos_to_path(CNshp['geometry'].to_list()))
[collection.set_clip_path(path_clip, transform=ax1.transData) for collection in cf.collections]
[collection.set_clip_path(path_clip, transform=ax1.transData) for collection in c1.collections]
[collection.set_clip_path(path_clip, transform=ax1.transData) for collection in c2.collections]

shp=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
ax1.add_geometries(shp.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=1.2,facecolor='none')
ax1.set_xticks([80,90,100,110,120,130])  #指定要显示的经纬度
ax1.set_yticks([20,25,30,35,40,45,50,55])
ax1.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(2))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(1))
cbar_ax=fig.add_axes([0.14,0.16,0.75,0.02])
cbar=plt.colorbar(cf,orientation='horizontal',cax=cbar_ax)
cbar.ax.tick_params(labelsize=9)


#%%保存青藏高原西部的shp文件
path = c2.collections[0].get_paths()[0] # 获取等值线的路径
coords = path.vertices

counter_line = LineString(coords) # 创建等值线的LineString几何对象
counter_line_gdf0 = gpd.GeoDataFrame(geometry=[counter_line])

filter_polygon = Polygon([(80, -90),(92, -90),(92, 90),(80, 90)])
counter_line=counter_line.intersection(filter_polygon) #截取等值线在26N-22.5N范围内的部分


coord=list(counter_line.coords) #获取3mm/day等值线的坐标点
coord.insert(0,(73,42))  #添加相应的经线和纬线对应的坐标点，使（等值线、经线、纬线）三者构成闭合图形
coord.insert(1,(92,42))
coord.insert(2,(92,33.5))
coord.append((73,27.5))
counter_line=Polygon(coord).buffer(0)
counter_line_gdf = gpd.GeoDataFrame(geometry=[counter_line])

CNgdf = gpd.read_file("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
TPgdf = gpd.read_file("D:/Python/untitled/青藏高原边界数据总集/TPBoundary_new(2021)/TPBoundary_new(2021).shp")
CNTPgdf = CNgdf.intersection(TPgdf) #提取中国边界与青藏高原的交集
CNTPgdf = CNTPgdf.intersection(counter_line_gdf) #提取上一交集与西北地区的交集

CNTPgdf.to_file('./data/TP_mask.shp') # 保存青藏高原西部shp文件

#%%保存西北地区边界文件（带青藏高原西部的和不带青藏高原西部的）
path = c2.collections[0].get_paths()[0] # 获取等值线的路径
coords = path.vertices

counter_line = LineString(coords) # 创建等值线的LineString几何对象
coord=list(counter_line.coords)[2:] #获取3mm/day等值线的坐标点，最东北边的尖尖去掉不要了
coord.insert(0,(72.5,52))  #添加相应的经线和纬线对应的坐标点，使（等值线、经线、纬线）三者构成闭合图形
coord.insert(1,(120, 52))
coord.append((72.5,27.5))
counter_line=Polygon(coord).buffer(0)
counter_line_gdf = gpd.GeoDataFrame(geometry=[counter_line]) #左上角边界是矩形，右下角是3mm/d等值线的多边形

CNgdf = gpd.read_file("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
TPmaskgdf = gpd.read_file("./data/TP_mask.shp")
NWCgdf=CNgdf.intersection(counter_line_gdf) #取多边形和中国地图的交集
NWC_TPgdf=NWCgdf.difference(TPmaskgdf) #取上面的交集与青藏高原西部的非集

NWCgdf.to_file('./data/NWC.shp') # 完整西北地区边界shp文件
NWC_TPgdf.to_file('./data/NWC_TP.shp') # 去除青藏高原西部的西北地区边界shp文件

#%%根据青藏高原西部shp得到mask数组
mask=np.zeros([I,J])*np.nan #用来确定西北边界的数组，需要3mm/day以下的正常区域设置为1，没有站点区域设置为缺测
for i in range(I):
    for j in range(J):
        if pre_copy[i,j]<=3:
            mask[i,j]=1
        if CNTPgdf.geometry.contains(Point(Lon[i,j],Lat[i,j]+1))[0]:
            mask[i,j]=np.nan #<3mm/d但位于青藏高原西部的不要

np.save('./data/maskXB.npy',mask) #西北地区mask数组（分辨率1x1）

#%%预览青藏高原西部shp文件的效果
lines=np.arange(0,7.1,0.5)

fig=plt.figure(figsize=(7,6))
ax1=plt.subplot(111,projection=ccrs.PlateCarree())
ax1.set_extent([72,137,17,55],crs=ccrs.PlateCarree())
# ax1.set_title('1961-2021年夏季降水量平均态',fontsize=15)
ax1.add_feature(cfeature.OCEAN,color='lightblue')
cf=ax1.contourf(lon,lat,pre_mean,lines,transform=ccrs.PlateCarree(),cmap=cmap,extend='max')
c1=ax1.contour(lon,lat,pre_copy,[3],transform=ccrs.PlateCarree(),colors='r',linewidths=0.5)
c2=ax1.contour(lon,lat,pre_copy,[3],transform=ccrs.PlateCarree(),colors='r',linewidths=1.5)

#只保留中国区域内的部分
CNshp=gpd.read_file("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
path_clip = Path.make_compound_path(*geos_to_path(CNshp['geometry'].to_list()))
[collection.set_clip_path(path_clip, transform=ax1.transData) for collection in cf.collections]
[collection.set_clip_path(path_clip, transform=ax1.transData) for collection in c1.collections]
[collection.set_clip_path(path_clip, transform=ax1.transData) for collection in c2.collections]

shp=Reader("D:/python/untitled/cnmap/china-shapefiles-master/shapefiles/china_without_islands.shp")
TP_mask=Reader("./data/TP_mask.shp")
ax1.add_geometries(TP_mask.geometries(),crs=ccrs.PlateCarree(),edgecolor='none',linewidths=0,facecolor='0.9')
ax1.add_geometries(shp.geometries(),crs=ccrs.PlateCarree(),edgecolor='k',linewidths=1.2,facecolor='none')
ax1.set_xticks([80,90,100,110,120,130])  #指定要显示的经纬度
ax1.set_yticks([20,25,30,35,40,45,50,55])
ax1.xaxis.set_major_formatter(LongitudeFormatter())  #刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(2))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(1))
cbar_ax=fig.add_axes([0.14,0.16,0.75,0.02])
cbar=plt.colorbar(cf,orientation='horizontal',cax=cbar_ax)
cbar.ax.tick_params(labelsize=9)

plt.savefig('./figure/Supplementary Figure 6.pdf', bbox_inches='tight')
