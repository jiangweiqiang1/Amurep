# -*- coding: utf-8 -*-
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
from utils.func import *
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
import networkx as nx
from model import *
import random
import torch
import csv
import logging
import numpy as np
import pickle
import json
import torch.nn.functional as F

# 获取当前时间戳作为日志文件名的一部分
log_file_name = f'logs/log_{int(time.time())}.log'
# 配置日志
logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':

    for dataSource in ['gowalla']:
        arg = {}

        start_time = time.time()

        arg['epoch'] = 50
        arg['beamSize'] = 100
        arg['embedding_dim'] = 1024
        arg['userEmbed_dim'] = 1024
        arg['hidden_dim'] = 512
        arg['classification_learning_rate'] = 0.0001
        arg['classification_batch'] = 32
        arg['dropout'] = 0.9
        arg['dataFolder'] = 'processedFiles'

        print()
        print(dataSource)
        print()
        print(arg)
        logging.info(dataSource)
        logging.info(arg)

        # ==================================Spatial Temporal graphs================================================
        arg['temporalGraph'] = nx.read_edgelist('data/' + arg['dataFolder'] + '/' + dataSource + '_temporal.edgelist',
                                                nodetype=int, create_using=nx.Graph())

        arg['spatialGraph'] = nx.read_edgelist('data/' + arg['dataFolder'] + '/' + dataSource + '_spatial.edgelist',
                                               nodetype=int, create_using=nx.Graph())

        arg['usersGraph'] = nx.read_edgelist('data/' + arg['dataFolder'] + '/' + dataSource + '_poipoi2.edgelist',
                                             nodetype=int, create_using=nx.Graph())
        # ==================================Spatial Temporal graphs================================================

        userFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_userCount.pickle'

        with open(userFileName, 'rb') as handle:
            arg['numUser'] = pickle.load(handle)

        print('Data loading')
        logging.info('1.Data loading')

        # ==================================geohash related data================================================
        for eachGeoHashPrecision in [6, 5, 4, 3, 2]:
            poi2geohashFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_poi2geohash' + '_' + str(
                eachGeoHashPrecision)
            poi2geohashFileName2 = 'data/' + arg['dataFolder'] + '/' + dataSource + '_poi2geohash2' + '_' + str(
                eachGeoHashPrecision)
            geohash2poiFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_geohash2poi' + '_' + str(
                eachGeoHashPrecision)
            geohash2IndexFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_geohash2Index' + '_' + str(
                eachGeoHashPrecision)

            with open(poi2geohashFileName + '.pickle', 'rb') as handle:
                arg['poi2geohash' + '_' + str(eachGeoHashPrecision)] = pickle.load(handle)
            with open(poi2geohashFileName2 + '.pickle', 'rb') as handle:
                arg['poi2geohash2' + '_' + str(eachGeoHashPrecision)] = pickle.load(handle)
            with open(geohash2poiFileName + '.pickle', 'rb') as handle:
                arg['geohash2poi' + '_' + str(eachGeoHashPrecision)] = pickle.load(handle)
            with open(geohash2IndexFileName + '.pickle', 'rb') as handle:
                arg['geohash2Index' + '_' + str(eachGeoHashPrecision)] = pickle.load(handle)

            arg['index2geoHash' + '_' + str(eachGeoHashPrecision)] = {v: k for k, v in arg[
                'geohash2Index' + '_' + str(eachGeoHashPrecision)].items()}

        beamSearchHashDictFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_beamSearchHashDict'
        with open(beamSearchHashDictFileName + '.pickle', 'rb') as handle:
            arg['beamSearchHashDict'] = pickle.load(handle)
        # 求KL散度的
        # 读取 POI_POIjiaohu.npy 文件
        # poi_poi_jiaohu = np.load('utils/POI_POIjiaohu2.npy')
        # # 将加载的矩阵转换为 PyTorch 张量
        # poi_poi_jiaohu_tensor = torch.tensor(poi_poi_jiaohu, dtype=torch.float).cuda()
        # # 将加载的矩阵添加到 arg 字典中
        # arg['POI_POIjiaohu'] = poi_poi_jiaohu_tensor

        poi_poi_2 = np.load('utils/POI_cu2_kong1.npy')
        poi_poi_2_tensor = torch.tensor(poi_poi_2, dtype=torch.float).cuda()
        arg['POI_POI2'] = poi_poi_2_tensor
        poi_poi_22 = torch.argmax(poi_poi_2_tensor, dim=1)

        poi_poi_3 = np.load('utils/POI_cu2_kong2.npy')
        poi_poi_3_tensor = torch.tensor(poi_poi_3, dtype=torch.float).cuda()
        arg['POI_POI3'] = poi_poi_3_tensor
        poi_poi_33 = torch.argmax(poi_poi_3_tensor, dim=1)

        poi_poi_4 = np.load('utils/POI_cu2_kong3.npy')
        poi_poi_4_tensor = torch.tensor(poi_poi_4, dtype=torch.float).cuda()
        arg['POI_POI4'] = poi_poi_4_tensor
        poi_poi_44 = torch.argmax(poi_poi_4_tensor, dim=1)

        poi_poi_5 = np.load('utils/POI_cu2_kong4.npy')
        poi_poi_5_tensor = torch.tensor(poi_poi_5, dtype=torch.float).cuda()
        arg['POI_POI5'] = poi_poi_5_tensor
        poi_poi_55 = torch.argmax(poi_poi_5_tensor, dim=1)

        poi_poi_6 = np.load('utils/POI_cu2_kong5.npy')
        poi_poi_6_tensor = torch.tensor(poi_poi_6, dtype=torch.float).cuda()
        arg['POI_POI6'] = poi_poi_6_tensor
        poi_poi_66 = torch.argmax(poi_poi_6_tensor, dim=1)

        # ==================================geohash related data================================================

        classification_dataset = classificationDataset(arg['numUser'], dataSource, arg)

        classification_dataloader = DataLoader(classification_dataset, batch_size=arg['classification_batch'],
                                               shuffle=True, pin_memory=True, num_workers=0)
        print('Data loaded')
        print('init model')
        logging.info('3.Data loaded')
        logging.info('4.init model')

        classification = hmt_grn(arg).float().cuda()
        # classification2 = zong(arg).float().cuda()

        classification_optim = Adam(classification.parameters(), lr=arg['classification_learning_rate'])

        print('init model done')
        logging.info('5.init model done')

        criterion = nn.NLLLoss(reduction='mean', ignore_index=0)

        nextGeoHashCriterion_2 = nn.NLLLoss(reduction='mean', ignore_index=0)
        nextGeoHashCriterion_3 = nn.NLLLoss(reduction='mean', ignore_index=0)
        nextGeoHashCriterion_4 = nn.NLLLoss(reduction='mean', ignore_index=0)
        nextGeoHashCriterion_5 = nn.NLLLoss(reduction='mean', ignore_index=0)
        nextGeoHashCriterion_6 = nn.NLLLoss(reduction='mean', ignore_index=0)
        # kl_loss2 = torch.nn.KLDivLoss(reduction="batchmean")
        # kl_loss3 = torch.nn.KLDivLoss(reduction="batchmean")
        # kl_loss4 = torch.nn.KLDivLoss(reduction="batchmean")
        # kl_loss5 = torch.nn.KLDivLoss(reduction="batchmean")
        # kl_loss6 = torch.nn.KLDivLoss(reduction="batchmean")
        # kl_losslinjie = torch.nn.KLDivLoss(reduction="batchmean")
        jiaocha2 = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        jiaocha3 = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        jiaocha4 = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        jiaocha5 = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        jiaocha6 = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        # 加载邻接矩阵 POI_POI.npy，并将数据类型转换为整数类型
        # POI_POI = torch.tensor(np.load('utils/poi_poi2.npy'), dtype=torch.float)
        POI_POI = torch.tensor(np.load('utils/cosine_similarity_matrix.npy'), dtype=torch.float)
        POI_POI = POI_POI.cuda()

        for epoch in range(1, arg['epoch'] + 1):
            avgLossDict = {}
            print()
            print('Epoch: ' + str(epoch))
            logging.info("----- START SAVING -----")
            logging.info("Epoch: %s", arg['epoch'])
            logging.info("Next POI Classification Avg Loss: ")

            avgLossDict['Next POI Classification'] = []

            classification_pbar = tqdm(classification_dataloader)

            classification_pbar.set_description('[' + dataSource + "_Classification-Epoch {}]".format(epoch))

            for x, user, y in classification_pbar:
                actualBatchSize = x.shape[0]  # 获取当前批次的实际样本数量 batch
                batchLoss = 0  # 批次损失
                # (batch,max_length)
                x_geoHash2 = LT([]).cuda()
                x_geoHash3 = LT([]).cuda()
                x_geoHash4 = LT([]).cuda()
                x_geoHash5 = LT([]).cuda()
                x_geoHash6 = LT([]).cuda()
                # 遍历批次下的每一个样本 batch
                for eachBatch in range(x.shape[0]):
                    # 用户访问过的POI列表
                    sample = x[eachBatch].tolist()  # 将每个样本转换为列表 poi序列转化为列表
                    # 对每个样本的地理哈希进行映射并构建对应维度的张量   1.每个区域下哈希值对应的索引  2.每个poi对应不同区域下的哈希
                    # 找到poi访问序列对应的区域索引，mappedGeohash列表31个，没有的补0 （先找POI对应的哈希值，在找区域哈希值对应的索引） x_geoHash2：(batch,max_length)
                    mappedGeohash = [arg['geohash2Index' + '_2'][arg['poi2geohash' + '_2'][i]] for i in sample]
                    x_geoHash2 = t.cat((x_geoHash2, LT(mappedGeohash).unsqueeze(0).cuda()), dim=0)

                    mappedGeohash = [arg['geohash2Index' + '_3'][arg['poi2geohash' + '_3'][i]] for i in sample]
                    x_geoHash3 = t.cat((x_geoHash3, LT(mappedGeohash).unsqueeze(0).cuda()), dim=0)

                    mappedGeohash = [arg['geohash2Index' + '_4'][arg['poi2geohash' + '_4'][i]] for i in sample]
                    x_geoHash4 = t.cat((x_geoHash4, LT(mappedGeohash).unsqueeze(0).cuda()), dim=0)

                    mappedGeohash = [arg['geohash2Index' + '_5'][arg['poi2geohash' + '_5'][i]] for i in sample]
                    x_geoHash5 = t.cat((x_geoHash5, LT(mappedGeohash).unsqueeze(0).cuda()), dim=0)

                    mappedGeohash = [arg['geohash2Index' + '_6'][arg['poi2geohash' + '_6'][i]] for i in sample]
                    x_geoHash6 = t.cat((x_geoHash6, LT(mappedGeohash).unsqueeze(0).cuda()), dim=0)

                # 输入为 批次下的第一序列、用户号、第二序列、poi所属区域的索引 大小都是(batch,max_length)
                input = (x, user, y, x_geoHash2, x_geoHash3, x_geoHash4, x_geoHash5, x_geoHash6)
                # 返回（32,31，每个poi的概率，每个区域类中的概率）
                logSoftmaxScores, nextgeohashPred_2, nextgeohashPred_3, nextgeohashPred_4, nextgeohashPred_5, nextgeohashPred_6, \
                    POI_cu2, POI_cu3, POI_cu4, POI_cu5, POI_cu6 = classification(input, 'train', arg, POI_POI, epoch,
                                                                                 dataSource)

                truth = LT(y.numpy()).cuda()

                # map truth to geohash
                truthDict = {}
                for eachGeoHashPrecision in [6, 5, 4, 3, 2]:
                    name = 'nextGeoHashTruth' + '_' + str(eachGeoHashPrecision)
                    behind = '_' + str(eachGeoHashPrecision)
                    truthDict[name] = LT([]).cuda()
                    for eachBatch in range(truth.shape[0]):
                        sample = truth[eachBatch].tolist()
                        mappedNextGeohashTruth = [arg['geohash2Index' + behind][arg['poi2geohash2' + behind][i]] for i
                                                  in sample]
                        truthDict[name] = t.cat((truthDict[name], LT(mappedNextGeohashTruth).unsqueeze(0).cuda()),
                                                dim=0)

                class_size = logSoftmaxScores.shape[2]

                classification_loss = criterion(logSoftmaxScores.view(-1, class_size), truth.view(-1))
                nextGeoHash_loss_2 = nextGeoHashCriterion_2(nextgeohashPred_2.view(-1, len(arg['geohash2Index_2'])),
                                                            truthDict['nextGeoHashTruth_2'].view(-1))
                nextGeoHash_loss_3 = nextGeoHashCriterion_3(nextgeohashPred_3.view(-1, len(arg['geohash2Index_3'])),
                                                            truthDict['nextGeoHashTruth_3'].view(-1))
                nextGeoHash_loss_4 = nextGeoHashCriterion_4(nextgeohashPred_4.view(-1, len(arg['geohash2Index_4'])),
                                                            truthDict['nextGeoHashTruth_4'].view(-1))
                nextGeoHash_loss_5 = nextGeoHashCriterion_5(nextgeohashPred_5.view(-1, len(arg['geohash2Index_5'])),
                                                            truthDict['nextGeoHashTruth_5'].view(-1))
                nextGeoHash_loss_6 = nextGeoHashCriterion_6(nextgeohashPred_6.view(-1, len(arg['geohash2Index_6'])),
                                                            truthDict['nextGeoHashTruth_6'].view(-1))
                # kl散度
                # a2 = kl_loss2(torch.log(softmax_POI_cu2_times_transpose + 1e-9), arg['POI_POI2'])
                # a3 = kl_loss3(torch.log(softmax_POI_cu3_times_transpose + 1e-9), arg['POI_POI3'])
                # a4 = kl_loss4(torch.log(softmax_POI_cu4_times_transpose + 1e-9), arg['POI_POI4'])
                # a5 = kl_loss5(torch.log(softmax_POI_cu5_times_transpose + 1e-9), arg['POI_POI5'])
                # a6 = kl_loss6(torch.log(softmax_POI_cu6_times_transpose + 1e-9), arg['POI_POI6'])
                # a7 = kl_losslinjie(torch.log(adj3359 + 1e-9), arg['POI_POIjiaohu'])
                b2 = jiaocha2(POI_cu2, poi_poi_22)  # 2个矩阵求
                b3 = jiaocha2(POI_cu3, poi_poi_33)
                b4 = jiaocha2(POI_cu4, poi_poi_44)
                b5 = jiaocha2(POI_cu5, poi_poi_55)
                b6 = jiaocha2(POI_cu6, poi_poi_66)

                # batchLoss = (classification_loss  +  nextGeoHash_loss_2 +  nextGeoHash_loss_3 + nextGeoHash_loss_4 + nextGeoHash_loss_5 + nextGeoHash_loss_6)  / 6 / actualBatchSize
                batchLoss = (
                                        classification_loss + b2 + b3 + b4 + b5 + b6 + nextGeoHash_loss_2 + nextGeoHash_loss_3 + nextGeoHash_loss_4 + nextGeoHash_loss_5 + nextGeoHash_loss_6) / 11 / actualBatchSize
                # batchLoss = classification_loss * actualBatchSize

                classification_optim.zero_grad()
                batchLoss.backward(retain_graph=False)
                classification_optim.step()
                classification_pbar.set_postfix(loss=classification_loss.item() / actualBatchSize)

                avgLossDict['Next POI Classification'].append(classification_loss.item() / actualBatchSize)

            avgLossDict['Next POI Classification'] = np.average(avgLossDict['Next POI Classification'])

            # 定义权重矩阵 W 的形状，并将数据类型转换为整数类型
            W_shape2, W_shape3, W_shape4, W_shape5, W_shape6 = (3359, 67), (67, 268), (268, 604), (604, 1071), (
            1071, 1812)
            # 创建一个空字典用于存储结果
            result_dict2, result_dict3, result_dict4, result_dict5, result_dict6 = {}, {}, {}, {}, {}
            # 获取每一行的最大值及其对应的列索引
            max_values2, max_indices2 = torch.max(POI_cu2, dim=1)
            max_values3, max_indices3 = torch.max(POI_cu3, dim=1)
            max_values4, max_indices4 = torch.max(POI_cu4, dim=1)
            max_values5, max_indices5 = torch.max(POI_cu5, dim=1)
            max_values6, max_indices6 = torch.max(POI_cu6, dim=1)

            unique_indices2 = set(max_indices2.tolist())  # 转换为列表再转换为集合
            unique_indices3 = set(max_indices3.tolist())
            unique_indices4 = set(max_indices4.tolist())
            unique_indices5 = set(max_indices5.tolist())
            unique_indices6 = set(max_indices6.tolist())
            # 输出唯一值的数量
            num_unique_indices2 = len(unique_indices2)
            num_unique_indices3 = len(unique_indices3)
            num_unique_indices4 = len(unique_indices4)
            num_unique_indices5 = len(unique_indices5)
            num_unique_indices6 = len(unique_indices6)
            print("Number of unique values in max_indices2:", num_unique_indices2)
            print("Number of unique values in max_indices3:", num_unique_indices3)
            print("Number of unique values in max_indices4:", num_unique_indices4)
            print("Number of unique values in max_indices5:", num_unique_indices5)
            print("Number of unique values in max_indices6:", num_unique_indices6)
            logging.info("Number of unique values in max_indices2: %d", num_unique_indices2)
            logging.info("Number of unique values in max_indices2: %d", num_unique_indices3)
            logging.info("Number of unique values in max_indices2: %d", num_unique_indices4)
            logging.info("Number of unique values in max_indices2: %d", num_unique_indices5)
            logging.info("Number of unique values in max_indices2: %d", num_unique_indices6)

            # 创建一个列表，包含所有需要处理的矩阵和字典
            matrices = [(POI_cu2, max_indices2, result_dict2, W_shape2),
                        (POI_cu3, max_indices3, result_dict3, W_shape3),
                        (POI_cu4, max_indices4, result_dict4, W_shape4),
                        (POI_cu5, max_indices5, result_dict5, W_shape5),
                        (POI_cu6, max_indices6, result_dict6, W_shape6)]
            # 遍历每个矩阵及其相关的字典
            for POI_cu, max_indices, result_dict, W_shape in matrices:
                # 将相同列索引的行号组合在一起
                for i in range(POI_cu.shape[0]):
                    index = (max_indices[i] + 1).item()  # 获取列索引的整数值
                    if index not in result_dict:
                        result_dict[index] = []  # 创建新的键值对
                    result_dict[index].append(i + 1)  # 将行号添加到对应的列表中
                # 检查每个类别是否至少有一个 POI
                for idx in range(W_shape[1]):
                    if idx + 1 not in result_dict:
                        result_dict[idx + 1] = []

            if epoch == 1:
                print(POI_cu2.shape, POI_cu3.shape, POI_cu4.shape, POI_cu5.shape, POI_cu6.shape)
                logging.info("Shape of POI_cu2: %s", POI_cu2.shape)
                logging.info("Shape of POI_cu3: %s", POI_cu3.shape)
                logging.info("Shape of POI_cu4: %s", POI_cu4.shape)
                logging.info("Shape of POI_cu5: %s", POI_cu5.shape)
                logging.info("Shape of POI_cu6: %s", POI_cu6.shape)
            # 将结果字典转换为列表形式并按列索引排序
            result_list2 = sorted(result_dict2.items(), key=lambda x: x[0])
            result_list3 = sorted(result_dict3.items(), key=lambda x: x[0])
            result_list4 = sorted(result_dict4.items(), key=lambda x: x[0])
            result_list5 = sorted(result_dict5.items(), key=lambda x: x[0])
            result_list6 = sorted(result_dict6.items(), key=lambda x: x[0])
            '''取每一个POI属于哪一簇'''
            # 假设result_lists是一个字典，包含所有层次聚类结果的列表
            result_lists = {
                'cu2': (result_list2, 'POI_cu2.csv'),
                'cu3': (result_list3, 'POI_cu3.csv'),
                'cu4': (result_list4, 'POI_cu4.csv'),
                'cu5': (result_list5, 'POI_cu5.csv'),
                'cu6': (result_list6, 'POI_cu6.csv')
            }
            # 确保目录存在
            os.makedirs('utils/zidong', exist_ok=True)

            for cluster_name, (result_list, filename) in result_lists.items():
                # 构造完整的文件路径
                file_path = os.path.join('utils/zidong', filename)

                # 打开文件并写入内容
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for _, row_indices in result_list:
                        writer.writerow(row_indices)
            '''1.开始生成gowalla_geohash2poi_N'''
            '''1.开始生成gowalla_geohash2poi_N'''
            '''1.开始生成gowalla_geohash2poi_N'''
            # 定义CSV文件路径和对应Pickle文件路径
            files = [
                ('utils/zidong/POI_cu2.csv', 'data/processedFiles/gowalla_geohash2poi_2.pickle', 1),
                ('utils/zidong/POI_cu3.csv', 'data/processedFiles/gowalla_geohash2poi_3.pickle', 68),
                ('utils/zidong/POI_cu4.csv', 'data/processedFiles/gowalla_geohash2poi_4.pickle', 336),
                ('utils/zidong/POI_cu5.csv', 'data/processedFiles/gowalla_geohash2poi_5.pickle', 940),
                ('utils/zidong/POI_cu6.csv', 'data/processedFiles/gowalla_geohash2poi_6.pickle', 2011)
            ]
            # 遍历每个文件路径和偏移量，进行处理
            for csv_file, pickle_file, start_index in files:
                clusters = {}
                with open(csv_file, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for idx, row in enumerate(reader, start=start_index):
                        clusters[str(idx)] = [int(x) for x in row]
                # 添加键为0、值为空列表的键值对
                clusters[0] = []
                # 保存为 Pickle 文件
                with open(pickle_file, 'wb') as f:
                    pickle.dump(clusters, f)
            '''2.开始生成gowalla_poi2geohash_N'''
            '''2.开始生成gowalla_poi2geohash_N'''
            '''2.开始生成gowalla_poi2geohash_N'''
            '''编号：几个POI ---> POI:编号'''
            # 定义要处理的数字列表
            numbers = [2, 3, 4, 5, 6]
            for number in numbers:
                input_filename = f'data/processedFiles/gowalla_geohash2poi_{number}.pickle'
                output_filename = f'data/processedFiles/gowalla_poi2geohash_{number}.pickle'

                # 读取输入文件
                with open(input_filename, 'rb') as f:
                    clusters = pickle.load(f)

                # 生成反向字典
                reversed_clusters = {}
                for key, values_list in clusters.items():
                    if not isinstance(values_list, (list, tuple)):
                        print(f"Non-iterable value for key {key}: {values_list}")
                        continue  # 跳过非迭代的值
                    for value in values_list:
                        reversed_clusters[str(value)] = int(key)
                # 排序反向字典
                sorted_reversed_clusters = dict(sorted(reversed_clusters.items(), key=lambda x: int(x[0])))
                # 将 key 转换为 int，value 转换为 str，并添加特殊条目
                converted_clusters = {int(key): str(value) for key, value in sorted_reversed_clusters.items()}
                converted_clusters[0] = 0
                # 写入输出文件
                with open(output_filename, 'wb') as f:
                    pickle.dump(converted_clusters, f)

            # =================geohash related data  三个哈希数据  1.每一个POI对应的哈希值 2.每一个区域分为多少个哈希 3.每个哈希下有几个兴趣点=======================
            for eachGeoHashPrecision in [6, 5, 4, 3, 2]:
                poi2geohashFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_poi2geohash' + '_' + str(
                    eachGeoHashPrecision)
                geohash2poiFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_geohash2poi' + '_' + str(
                    eachGeoHashPrecision)
                with open(poi2geohashFileName + '.pickle', 'rb') as handle:
                    arg['poi2geohash' + '_' + str(eachGeoHashPrecision)] = pickle.load(handle)
                with open(geohash2poiFileName + '.pickle', 'rb') as handle:
                    arg['geohash2poi' + '_' + str(eachGeoHashPrecision)] = pickle.load(handle)

            # 加载贪婪算法编码
            data_list = []
            start_values = [(1, 68), (68, 336), (336, 940), (940, 2011)]

            for n, (start_1, start_2) in enumerate(start_values, start=2):
                filename_1 = f'utils/zidong/POI_cu{n}.csv'
                filename_2 = f'utils/zidong/POI_cu{n + 1}.csv'

                clusters_0_n_1 = {idx: [int(point) for point in row] for idx, row in
                                  enumerate(csv.reader(open(filename_1, 'r', encoding='utf-8')), start=start_1)}
                clusters_0_n = {idx: [int(point) for point in row] for idx, row in
                                enumerate(csv.reader(open(filename_2, 'r', encoding='utf-8')), start=start_2)}

                associations = {
                    str(cluster_id_0_n_1): [str(cluster_id_0_n) for cluster_id_0_n, points_0_n in clusters_0_n.items()
                                            if any(point in points_0_n_1 for point in points_0_n)]
                    for cluster_id_0_n_1, points_0_n_1 in clusters_0_n_1.items()
                }

                with open(f'utils/zidong/gowall_zidong{n}_{n + 1}.json', 'w') as json_file:
                    json.dump(associations, json_file)

                data_list.append(associations)

            # 创建新的数据结构
            new_data = {f'{i}_{i + 1}': data for i, data in enumerate(data_list, start=2)}
            with open('data/processedFiles/gowalla_beamSearchHashDict.pickle', 'wb') as file:
                pickle.dump(new_data, file)
            # 更新自动学习的beamSearchHashDict字典
            beamSearchHashDictFileName = f"data/{arg['dataFolder']}/{dataSource}_beamSearchHashDict.pickle"
            with open(beamSearchHashDictFileName, 'rb') as handle:
                arg['beamSearchHashDict'] = pickle.load(handle)

            print('Next POI Classification Avg Loss: ' + str(avgLossDict['Next POI Classification']))
            logging.info('Epoch: %s', epoch)
            logging.info('Next POI Classification Avg Loss: %s', str(avgLossDict['Next POI Classification']))

            print()
            sys.stdout.flush()
            print()
            print("----- END SAVING  : %s seconds ---" % (time.time() - start_time))
            logging.info('----- END SAVING  : %s seconds ---', (time.time() - start_time))
            print()
            if epoch % 2 == 0:
                print()
                print('Epoch ' + str(epoch) + ' Evaluation Start!')
                print()
                model = classification
                arg['novelEval'] = False
                evaluate(model, dataSource, arg, POI_POI, epoch)
                arg['novelEval'] = True
                evaluate(model, dataSource, arg, POI_POI, epoch)
            sys.stdout.flush()
