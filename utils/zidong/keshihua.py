# import matplotlib as mpl
# mpl.use('agg')
# import csv
# import pandas as pd
# # import matplotlib
# import matplotlib.pyplot as plt
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
#
# # 读取POI_cu2.csv文件
# with open('POI_cu2.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     poi_cu2_data = list(reader)
#
# # 读取gowalla_allData.csv文件
# gowalla_df = pd.read_csv('gowalla_allData.csv', usecols=['lat', 'lon', 'POI'])
#
# # 设置颜色映射，每个簇使用一种颜色
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#
# # 遍历POI_cu2.csv中的每一行
# for i, row in enumerate(poi_cu2_data):
#     # 为每个簇选择一种颜色
#     color = colors[i % len(colors)]
#
#     # 遍历当前行中的每个POI
#     for poi in row:
#         # 在gowalla_df中找到当前POI的经纬度信息
#         poi_info = gowalla_df[gowalla_df['POI'] == int(poi)]
#
#         # 如果找到了对应的POI信息，则绘制一个点
#         if not poi_info.empty:
#             plt.scatter(poi_info['lon'], poi_info['lat'], color=color, label=f'Cluster {i}')
#
# # 添加图例和标签
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('POI Clusters Visualization')
# plt.legend()
# plt.show()
# # plt.savefig("/autodl-fs/HMT/test.jpg")

import  pickle
import csv
with open('gowalla_geohash2poi_6.pickle', 'rb') as file: # gowalla_geohash2poi_6 0:[]  gowalla_poi2geohash_6 0:0
    data = pickle.load(file)
    print(data)

# clusters2 = {}
# with open('POI_cu4.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     for idx, row in enumerate(reader, start=1):  # 1
#         clusters2[str(idx)] = [int(x) for x in row]
# # 添加键为0、值为空列表的键值对
# clusters2[0] = []
# # 保存为 Pickle 文件
# with open('gowalla_geohash2poi_4.pickle', 'wb') as f:
#     pickle.dump(clusters2, f)
# with open('gowalla_geohash2poi_4.pickle', 'rb') as file:
#     data = pickle.load(file)
#     print(data)