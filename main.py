import numpy as np
import argparse
import random
import torch
import torch.optim as optim
import os
import heapq
import math
import time

from model import PatentSubgraph, PatentSubgraphPlus
from parser_ps import *

# 数据加载函数
def load_data(folder_path, meta_type):
    data = {
        'train_path': [],
        'train_neg_path': [],
        'test_path': [],
        'test_neg_path': [],
        'meta_type': meta_type
    }
    for file_name in ['train_path.txt', 'train_neg_path.txt', 'test_path.txt', 'test_neg_path.txt']:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            paths = []
            for line in file:
                items = line.strip().split('\t')
                company_pid = items[0].split(',')
                company = int(company_pid[0])
                pid = int(company_pid[1])
                num_paths = int(items[1])
                path_data = [list(map(int, path.split())) for path in items[2:]]
                paths.append((company, pid, num_paths, path_data))
            data[file_name[:-4]] = paths
    return data

# 读取映射文件
def read_mapping_file(file_path):
    mapping = {}
    line_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            line_count += 1
            values = line.strip().split(',')
            field = values[0]
            index = int(values[1])
            mapping[field] = index
    return mapping, line_count

def read_connection_file(file_path):
    connections = {}
    with open(file_path, 'r') as file:
        for line in file:
            id1, id2 = line.strip().split(',')
            id1 = int(id1)
            id2 = int(id2)
            if id1 in connections:
                connections[id1].append(id2)
            else:
                connections[id1] = [id2]
    return connections

def precision_at_k(top_score_dict, k, test_dict):
    precision = 0.0
    skipped_user_num = 0
    for user in top_score_dict:
        test_user = False
        for key in list(test_dict.keys()):
            if user == key[0]:
                test_user = True
                break
        
        if test_user == False:
            skipped_user_num = skipped_user_num + 1
            continue

        candidate_item = top_score_dict[user]
        candidate_size = len(candidate_item)
        hit = 0
        min_len = min(candidate_size, k)
        for i in range(min_len):
            if (user, candidate_item[i]) in test_dict:
                hit = hit + 1
        hit_ratio = float(hit / min_len)
        precision += hit_ratio
    
    precision = precision / (len(top_score_dict) - skipped_user_num)

    return precision

def recall_at_k(top_score_dict, k, test_dict):
    recall = 0.0
    skipped_user_num = 0
    for user in top_score_dict:
        candidate_item = top_score_dict[user]
        candidate_size = len(candidate_item)
        hit = 0
        min_len = min(candidate_size, k)
        for i in range(min_len):
            if (user, candidate_item[i]) in test_dict:
                hit = hit + 1

        positive_num = 0
        for key in list(test_dict.keys()):
            if user == key[0]:
                positive_num += 1

        if positive_num == 0:
            skipped_user_num = skipped_user_num + 1
            continue

        hit_ratio = float(hit / positive_num)
        recall += hit_ratio
    
    recall = recall / (len(top_score_dict) - skipped_user_num)

    return recall

def ndcg_at_k(top_score_dict, k, test_dict):
    ndcg = 0.0
    skipped_user_num = 0    
    for user in top_score_dict:
        candidate_item = top_score_dict[user]
        candidate_size = len(candidate_item)
        min_len = min(candidate_size, k)

        positive_num = 0
        for key in list(test_dict.keys()):
            if user == key[0]:
                positive_num += 1

        if positive_num == 0:
            skipped_user_num = skipped_user_num + 1
            continue

        dcg = 0.0
        idcg = 0.0
        for i in range(min_len):
            if (user, candidate_item[i]) in test_dict:
                dcg = dcg + (2 ** 1 - 1) / math.log2(i + 2)
            else:
                dcg = dcg + (2 ** 0 - 1) / math.log2(i + 2)
            
            if i <= positive_num - 1:
                idcg = idcg + (2 ** 1 - 1) / math.log2(i + 2)
            else:
                idcg = idcg + (2 ** 0 - 1) / math.log2(i + 2)
        
        if idcg == 0:
            idcg = float('inf')
        
        ndcg = ndcg + float(dcg / idcg)
    
    ndcg = ndcg / (len(top_score_dict) - skipped_user_num)

    return ndcg

if __name__ == '__main__':
    args = parse_args()
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.dataset == 'patent_2425':
        # 读取各个元路径
        cpfp_folder = 'data/patent_2425/path/cpfp'
        cplp_folder = 'data/patent_2425/path/cplp'

        # 读取cpfp文件夹中的数据
        cpfp_data = load_data(cpfp_folder, 'cpfp')

        # 读取cplp文件夹中的数据
        cplp_data = load_data(cplp_folder, 'cplp')

        data_list = [cpfp_data, cplp_data]

        # 映射文件路径
        second_patentee_file = 'data/second_patentee2index.txt'
        pid_file = 'data/pid2index.txt'
        first_patentee_file = 'data/first_patentee2index.txt'
        ipcClass_file = 'data/ipcClass2index.txt'
        industry_file = 'data/industry2index.txt'

        # 读取second_patentee2index.txt
        second_patentee_mapping, second_patentee_num = read_mapping_file(second_patentee_file)

        # 读取pid2index.txt
        pid_mapping, pid_num = read_mapping_file(pid_file)

        # 读取first_patentee2index.txt
        first_patentee_mapping, first_patentee_num = read_mapping_file(first_patentee_file)

        # 读取ipcClass2index.txt
        ipcClass_mapping, ipcClass_num = read_mapping_file(ipcClass_file)

        # 读取industry2index.txt
        industry_mapping, industry_num = read_mapping_file(industry_file)

        # 连接关系文件路径
        company_industry_file = 'data/company_industry_index.txt'
        pid_first_patentee_file = 'data/pid_first_patentee_index.txt'
        pid_ipcClass_file = 'data/pid_ipcClass_index.txt'
        company_patent_file = 'data/cp_train.txt'

        company_industry_connection = read_connection_file(company_industry_file)
        company_patent_connection = read_connection_file(company_patent_file)

        pid_first_patentee_connection = read_connection_file(pid_first_patentee_file)
        pid_ipcClass_connection = read_connection_file(pid_ipcClass_file)
        patent_company_connection = {}
        with open(company_patent_file, 'r') as file:
            for line in file:
                company, patent = line.strip().split(',')
                company = int(company)
                patent = int(patent)
                if patent in patent_company_connection:
                    patent_company_connection[patent].append(company)
                else:
                    patent_company_connection[patent] = [company]

    else:
        # 读取各个元路径
        cpfp_folder = 'data/'+ args.dataset +'/path/cpfp'
        cpsp_folder = 'data/'+ args.dataset +'/path/cpsp'
        cifp_folder = 'data/'+ args.dataset +'/path/cifp'

        # 读取cpfp文件夹中的数据
        cpfp_data = load_data(cpfp_folder, 'cpfp')

        # 读取cplp文件夹中的数据
        cpsp_data = load_data(cpsp_folder, 'cpsp')

        cifp_data = load_data(cifp_folder, 'cifp')

        data_list = [cpfp_data, cpsp_data, cifp_data]

        # 映射文件路径
        pid_file = 'data/'+ args.dataset +'/pid2index.txt'
        patentee_file = 'data/'+ args.dataset +'/patentee2index.txt'
        ipcSubClass_file = 'data/'+ args.dataset +'/ipcSubClass2index.txt'
        industry_file = 'data/'+ args.dataset +'/industry2index.txt'
        appDate_file = 'data/'+ args.dataset +'/appDate2index.txt'

        # 读取pid2index.txt
        pid_mapping, pid_num = read_mapping_file(pid_file)

        # 读取patentee2index.txt
        patentee_mapping, patentee_num = read_mapping_file(patentee_file)

        # 读取ipcSubClass2index.txt
        ipcSubClass_mapping, ipcSubClass_num = read_mapping_file(ipcSubClass_file)

        # 读取industry2index.txt
        industry_mapping, industry_num = read_mapping_file(industry_file)

        # 读取appDate2index.txt
        appDate_mapping, appDate_num = read_mapping_file(appDate_file)

        # 连接关系文件路径
        company_industry_file = 'data/'+ args.dataset +'/company_industry_index.txt'
        pid_first_patentee_file = 'data/'+ args.dataset +'/pid_first_patentee_index.txt'
        pid_ipcSubClass_file = 'data/'+ args.dataset +'/pid_ipcSubClass_index.txt'
        company_patent_file = 'data/'+ args.dataset +'/cp_train.txt'
        pid_appDate_file = 'data/'+ args.dataset +'/pid_appDate_index.txt' 

        company_industry_connection = read_connection_file(company_industry_file)
        company_patent_connection = read_connection_file(company_patent_file)

        pid_first_patentee_connection = read_connection_file(pid_first_patentee_file)
        pid_ipcSubClass_connection = read_connection_file(pid_ipcSubClass_file)
        pid_appDate_connection = read_connection_file(pid_appDate_file)
        patent_company_connection = {}
        with open(company_patent_file, 'r') as file:
            for line in file:
                company, patent = line.strip().split(',')
                company = int(company)
                patent = int(patent)
                if patent in patent_company_connection:
                    patent_company_connection[patent].append(company)
                else:
                    patent_company_connection[patent] = [company]        

    # 生成 train_dict
    train_dict = {}
    for data in data_list:
        meta_type = data['meta_type']
        train_paths = data['train_path']

        for i in range(len(train_paths)):
            company, pid, num_paths, path_data = train_paths[i]
            if (company, pid) not in train_dict:
                train_dict[(company, pid)] = {}
            train_dict[(company, pid)][meta_type] = path_data

    # 生成 train_neg_dict
    train_neg_dict = {}
    for data in data_list:
        meta_type = data['meta_type']
        train_neg_paths = data['train_neg_path']

        for i in range(len(train_neg_paths)):
            neg_company, neg_pid, neg_num_paths, neg_path_data = train_neg_paths[i]
            if (neg_company, neg_pid) not in train_neg_dict:
                train_neg_dict[(neg_company, neg_pid)] = {}
            train_neg_dict[(neg_company, neg_pid)][meta_type] = neg_path_data

    # 生成 test_dict
    test_dict = {}
    for data in data_list:
        meta_type = data['meta_type']
        test_paths = data['test_path']

        for i in range(len(test_paths)):
            company, pid, num_paths, path_data = test_paths[i]
            if (company, pid) not in test_dict:
                test_dict[(company, pid)] = {}
            test_dict[(company, pid)][meta_type] = path_data

    # 生成 test_neg_dict
    test_neg_dict = {}
    for data in data_list:
        meta_type = data['meta_type']
        test_neg_paths = data['test_neg_path']

        for i in range(len(test_neg_paths)):
            neg_company, neg_pid, neg_num_paths, neg_path_data = test_neg_paths[i]
            if (neg_company, neg_pid) not in test_neg_dict:
                test_neg_dict[(neg_company, neg_pid)] = {}
            test_neg_dict[(neg_company, neg_pid)][meta_type] = neg_path_data

    if args.dataset == 'patent_2425':
        num_list = [second_patentee_num, pid_num, first_patentee_num, ipcClass_num, industry_num]
        connection_list = [company_industry_connection, company_patent_connection, pid_first_patentee_connection, pid_ipcClass_connection, patent_company_connection]
        # 定义连接关系字典
        link_map = {('c','i'): connection_list[0], ('c','p'): connection_list[1], ('p','f'): connection_list[2], ('p','l'): connection_list[3], ('p','c'): connection_list[4]}
    else:
        num_list = [pid_num, patentee_num, ipcSubClass_num, industry_num, appDate_num]
        connection_list = [company_industry_connection, company_patent_connection, pid_first_patentee_connection, pid_ipcSubClass_connection, pid_appDate_connection]
        # 定义连接关系字典
        link_map = {('c','i'): connection_list[0], ('c','p'): connection_list[1], ('p','f'): connection_list[2], ('p','s'): connection_list[3]}        

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型实例
    if args.dataset == 'patent_2425': 
        model = PatentSubgraph(args, num_list, connection_list, device)
    else:
        model = PatentSubgraphPlus(args, num_list, connection_list, device)        
    if torch.cuda.is_available():
        model = model.cuda()

    # 定义优化器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay = args.l2_weight)

    result_path = 'result/'+ args.dataset +'/results_23.txt'

    fw_results = open(result_path, 'a+')
    
    line = 'seed: ' + str(args.seed) + '\n'
    fw_results.write(line)
    line = 'dataset: ' + args.dataset + '\n'
    fw_results.write(line)    
    line = 'epoch: ' + str(args.epoch) + '\n'
    fw_results.write(line)
    line = 'node_emb_size: ' + str(args.node_emb_size) + '\n'
    fw_results.write(line)
    line = 'hidden_emb_size: ' + str(args.hidden_emb_size) + '\n'
    fw_results.write(line)
    line = 'learning rate: ' + str(args.lr) + '\n'
    fw_results.write(line)
    line = 'l2_weight: ' + str(args.l2_weight) + '\n'
    fw_results.write(line)
    line = 'use subgraph: ' + str(args.use_subgraph) + '\n'
    fw_results.write(line)
    line = 'aggregator: ' + args.aggregator + '\n'
    fw_results.write(line)    
    line = 'fusion_type: ' + args.fusion_type + '\n'
    fw_results.write(line)
    line = 'merge_type:' + args.merge_type + '\n'
    fw_results.write(line)
    line = 'current time is: ' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '\n\n'
    fw_results.write(line)
    fw_results.close()    

    selected_k = 5 # When finding the best results, it is the k to be based on
    best_recall = 0.0
    best_epoch = -1

    # 模型训练
    num_epochs = args.epoch
    for epoch in range(num_epochs):
        model.train()
        for key, value in train_dict.items():
            company, pid = key
            path_data_dict = value

            label = torch.tensor([[1.0]])

            # 前向传播
            return_dict = model(path_data_dict, label, link_map)

            # 计算损失
            loss = return_dict['loss']

            # 清空梯度
            optimizer.zero_grad()            
            # 反向传播和优化
            loss.backward()
            optimizer.step()

        for key, value in train_neg_dict.items():
            company, pid = key
            path_data_dict = value

            label = torch.tensor([[0.0]])

            # 前向传播
            return_dict = model(path_data_dict, label, link_map)

            # 计算损失
            loss = return_dict['loss']

            # 清空梯度
            optimizer.zero_grad()
            # 反向传播和优化
            loss.backward()
            optimizer.step()

        fw_results = open(result_path, 'a+')
        line = 'Epoch: ' + str(epoch) + ' training done!\n'
        fw_results.write(line)
        line = 'current time is: ' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '\n\n'
        fw_results.write(line)
        fw_results.close()

        # 每5个epoch进行测试
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            score_dict = {}

            for key, value in test_dict.items():
                company, pid = key
                path_data_dict = value

                label = torch.tensor([[1.0]])

                # 前向传播
                with torch.no_grad():
                    return_dict = model(path_data_dict, label, link_map)

                # 获取预测值
                prediction = return_dict['prediction'].cpu().numpy()

                if company not in score_dict:
                    score_dict.update({company:{pid:prediction}})
                else:
                    score_dict[company].update({pid:prediction})


            for key, value in test_neg_dict.items():
                company, pid = key
                path_data_dict = value

                label = torch.tensor([[0.0]])

                # 前向传播
                with torch.no_grad():
                    return_dict = model(path_data_dict, label, link_map)

                # 获取预测值
                prediction = return_dict['prediction'].cpu().numpy()

                if company not in score_dict:
                    score_dict.update({company:{pid:prediction}})
                else:
                    score_dict[company].update({pid:prediction})
            
            top_score_dict = {}
            for company in score_dict:
                if len(score_dict[company]) > 1:
                    item_score_list = score_dict[company]
                    l = len(item_score_list)
                    top_item_list = heapq.nlargest(l, item_score_list, key=item_score_list.get)
                    top_score_dict.update({company:top_item_list})

            precision_1 = precision_at_k(top_score_dict, 1, test_dict)
            precision_5 = precision_at_k(top_score_dict, 5, test_dict)
            precision_10 = precision_at_k(top_score_dict, 10, test_dict)
            precision_20 = precision_at_k(top_score_dict, 20, test_dict)

            recall_1 = recall_at_k(top_score_dict, 1, test_dict)
            recall_5 = recall_at_k(top_score_dict, 5, test_dict)
            recall_10 = recall_at_k(top_score_dict, 10, test_dict)
            recall_20 = recall_at_k(top_score_dict, 20, test_dict)

            ndcg_1 = ndcg_at_k(top_score_dict, 1, test_dict)
            ndcg_5 = ndcg_at_k(top_score_dict, 5, test_dict)
            ndcg_10 = ndcg_at_k(top_score_dict, 10, test_dict)
            ndcg_20 = ndcg_at_k(top_score_dict, 20, test_dict)

            if recall_5 > best_recall:
                best_recall = recall_5
                best_epoch = epoch

            fw_results = open(result_path, 'a+')
            line = 'Epoch: ' + str(epoch) + '\n'
            fw_results.write(line)
            line = 'precision_1: ' + str(precision_1) + '\n'    
            fw_results.write(line)
            line = 'precision_5: ' + str(precision_5) + '\n'    
            fw_results.write(line)
            line = 'precision_10: ' + str(precision_10) + '\n'    
            fw_results.write(line)
            line = 'precision_20: ' + str(precision_20) + '\n'    
            fw_results.write(line)
            line = 'recall_1: ' + str(recall_1) + '\n'    
            fw_results.write(line)
            line = 'recall_5: ' + str(recall_5) + '\n'    
            fw_results.write(line)
            line = 'recall_10: ' + str(recall_10) + '\n'    
            fw_results.write(line)
            line = 'recall_20: ' + str(recall_20) + '\n'    
            fw_results.write(line)
            line = 'ndcg_1: ' + str(ndcg_1) + '\n'    
            fw_results.write(line)
            line = 'ndcg_5: ' + str(ndcg_5) + '\n'    
            fw_results.write(line)
            line = 'ndcg_10: ' + str(ndcg_10) + '\n'    
            fw_results.write(line)
            line = 'ndcg_20: ' + str(ndcg_20) + '\n'    
            fw_results.write(line)
            line = 'current time is: ' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '\n\n'
            fw_results.write(line)
            fw_results.close()

            model.train()
    
    fw_results = open(result_path, 'a+')
    line = 'best epoch: ' + str(best_epoch) + '\n'
    fw_results.write(line)
    line = 'running done!' + '\n'
    fw_results.write(line)
    fw_results.close()
    print('running done!')