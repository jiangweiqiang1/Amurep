import csv
import numpy as np
'''邻接列表转化为邻接矩阵'''
# 初始化全零矩阵
# poi_poi = np.zeros((3359, 3359), dtype=int)
# # 读取 gowalla_spatial.edgelist 文件内容并更新邻接矩阵
# with open('gowalla_spatial.edgelist', 'r') as f:
#     for line in f:
#         # 提取起始节点和结束节点的索引
#         start, end, _ = line.strip().split()
#         # 将起始节点和结束节点的索引转换为整数
#         start, end = int(start), int(end)
#         # 在邻接矩阵中将对应的元素置为1
#         poi_poi[start-1, end-1] = 1
# # 保存邻接矩阵为 poi_poi.npy 文件
# np.save('gowalla_spatial.npy', poi_poi)
# print("Adjacency matrix saved as poi_poi.npy successfully.")


'''邻接矩阵变edgelist文件'''
# 加载 poi_poi 矩阵
poi_poi = np.load('gowalla_spatial.npy')
# 打开文件进行写入
with open('gowalla_spatial22.edgelist', 'w') as f:
    # 遍历 poi_poi 矩阵中的每个元素
    for i in range(poi_poi.shape[0]):
        for j in range(poi_poi.shape[1]):
            # 如果矩阵中的元素值大于0，则输出到文件中
            if poi_poi[i, j] > 0:
                # 将节点的起始节点和结束节点以及空字典输出到文件中
                f.write(f"{i+1} {j+1} {{}}\n")
print("Edgelist file generated successfully.")