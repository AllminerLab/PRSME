import torch
import torch.nn as nn

class PatentSubgraph(nn.Module):
    def __init__(self, args, num_list, connection_list, device):
        super(PatentSubgraph, self).__init__()
        self.embedding_dim = args.node_emb_size
        self.hidden_dim = args.hidden_emb_size
        self.num_list = num_list
        self.connection_list = connection_list
        self.device = device
        self.use_subgraph = args.use_subgraph
        self.aggregator = args.aggregator
        self.fusion_type = args.fusion_type
        self.merge_type = args.merge_type

        # 嵌入层
        self.company_embedding = nn.Embedding(num_list[0], self.embedding_dim)
        self.patent_embedding = nn.Embedding(num_list[1], self.embedding_dim)
        self.first_patentee_embedding = nn.Embedding(num_list[2], self.embedding_dim)
        self.ipcClass_embedding = nn.Embedding(num_list[3], self.embedding_dim)
        self.industry_embedding = nn.Embedding(num_list[4], self.embedding_dim)    

        # LSTM层
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)

        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # 融合子图表征时，注意力机制所用的MLP
        self.att_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

        # 获得子图表征时，采用gate形式时，aggregate gate所用的MLP
        self.agg_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Sigmoid()
        )

        # 获得子图表征时，采用gate形式时，filter gate所用的MLP
        self.filter_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Sigmoid()
        )

        # 获得子图表征时，采用gate形式时，最后所用的激活函数
        self.act = nn.LeakyReLU(negative_slope = 0.2)

        # 融合多路径表征时，采用的MLP
        self.merge_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim + self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # BCE损失函数
        self.bce_loss = nn.BCELoss()

        # 调用compute_subgraph_representation函数并将结果存储为模型内部属性
        #self.patent_subgraph_representation, self.company_subgraph_representation = self.compute_subgraph_representation()

    def compute_subgraph_representation(self):

        company_industry_connection = self.connection_list[0]
        company_patent_connection = self.connection_list[1]
        pid_first_patentee_connection = self.connection_list[2]
        pid_ipcClass_connection = self.connection_list[3]
        patent_company_connection = self.connection_list[4]

        # 存储专利子图和公司子图的嵌入向量
        self.patent_subgraph_representation = {}
        self.company_subgraph_representation = {}

        # 计算专利子图的嵌入向量
        for patent in range(self.num_list[1]):
            first_patentee_embeddings = self.first_patentee_embedding(torch.tensor(pid_first_patentee_connection[patent]).to(self.device))
            ipcClass_embeddings = self.ipcClass_embedding(torch.tensor(pid_ipcClass_connection[patent]).to(self.device))

            # 处理可能的字典缺失情况
            if patent in patent_company_connection:
                company_embeddings = self.company_embedding(torch.tensor(patent_company_connection[patent]).to(self.device))
                attribute_embeddings = torch.cat([first_patentee_embeddings, ipcClass_embeddings, company_embeddings], dim=0)
            else:
                attribute_embeddings = torch.cat([first_patentee_embeddings, ipcClass_embeddings], dim=0)

            patent_embedding = self.patent_embedding(torch.tensor(patent).to(self.device))
            patent_subgraph_embedding = patent_embedding + torch.mean(attribute_embeddings, dim=0)
            self.patent_subgraph_representation[patent] = patent_subgraph_embedding

        # 计算公司子图的嵌入向量
        for company in range(self.num_list[0]):
            industry_embeddings = self.industry_embedding(torch.tensor(company_industry_connection[company]).to(self.device))
            patent_embeddings = self.patent_embedding(torch.tensor(company_patent_connection[company]).to(self.device))
            attribute_embeddings = torch.cat([industry_embeddings, patent_embeddings], dim=0)

            company_embedding = self.company_embedding(torch.tensor(company).to(self.device))
            company_subgraph_embedding = company_embedding + torch.mean(attribute_embeddings, dim=0)
            self.company_subgraph_representation[company] = company_subgraph_embedding


    def forward(self, path_data_dict, label, link_map):
        # 定义一个列表存储融合路径表征
        fused_path_representations = []
        path_embeddings = []
        company_industry_connection = self.connection_list[0]
        company_patent_connection = self.connection_list[1]
        pid_first_patentee_connection = self.connection_list[2]
        pid_ipcClass_connection = self.connection_list[3]
        patent_company_connection = self.connection_list[4]

        # 处理路径集合
        for meta_type, path_data in path_data_dict.items():            

            # 将路径中的实体索引序列转换为嵌入序列
            for entity_indices in path_data:
                if meta_type == "cpfp":
                    entity_embeddings = torch.stack([
                        self.company_embedding(torch.tensor(entity_indices[0]).to(self.device)),
                        self.patent_embedding(torch.tensor(entity_indices[1]).to(self.device)),
                        self.first_patentee_embedding(torch.tensor(entity_indices[2]).to(self.device)),
                        self.patent_embedding(torch.tensor(entity_indices[3]).to(self.device))
                    ], dim=0)
                    # print(entity_embeddings.size())
                    # raise ValueError("For test.")

                    if self.use_subgraph:
                    # 更新节点嵌入
                        for i in range(1, len(entity_embeddings) - 1):
                            entity_embedding = entity_embeddings[i]

                            # 判断节点与company的连接关系
                            if (meta_type[0], meta_type[i]) in link_map:
                                connection_dict = link_map[(meta_type[0], meta_type[i])]
                                if entity_indices[i] in connection_dict[entity_indices[0]]:
                                    # 计算company子图表征
                                    industry_embeddings = self.industry_embedding(torch.tensor(company_industry_connection[entity_indices[0]]).to(self.device))
                                    patent_embeddings = self.patent_embedding(torch.tensor(company_patent_connection[entity_indices[0]]).to(self.device))
                                    attribute_embeddings = torch.cat([industry_embeddings, patent_embeddings], dim=0)
                                    company_embedding = self.company_embedding(torch.tensor(entity_indices[0]).to(self.device))
                                    if self.aggregator == 'avg':
                                        company_subgraph_embedding = company_embedding + torch.mean(attribute_embeddings, dim=0)
                                    elif self.aggregator == 'ewp':
                                        info_sum = torch.zeros(self.embedding_dim).to(self.device)
                                        for t in range(attribute_embeddings.size(0)):
                                            attribute_emb = attribute_embeddings[t]
                                            info = torch.mul(attribute_emb, company_embedding)
                                            info_sum = info_sum + info
                                        
                                        company_subgraph_embedding = company_embedding + info_sum
                                    
                                    elif self.aggregator == 'gate':
                                        # aggregate gate
                                        agg_representation = torch.zeros(self.embedding_dim).to(self.device)
                                        for t in range(attribute_embeddings.size(0)):
                                            attribute_emb = attribute_embeddings[t]
                                            com_cat_attr = torch.cat([company_embedding, attribute_emb]) # [dim * 2]
                                            agg_gate = self.agg_mlp(com_cat_attr)
                                            agg_representation = agg_representation + torch.mul(attribute_emb, agg_gate)
                                        
                                        agg_representation = torch.div(agg_representation, attribute_embeddings.size(0))

                                        # filter gate
                                        avg_neighbor = torch.mean(attribute_embeddings, dim=0)
                                        com_cat_avg_neighbor = torch.cat([company_embedding, avg_neighbor])
                                        filter_gate = self.filter_mlp(com_cat_avg_neighbor)
                                        all_one = torch.ones(self.embedding_dim).to(self.device)
                                        filter_representation = torch.mul(company_embedding, (all_one - filter_gate))

                                        company_subgraph_embedding = self.act(filter_representation + agg_representation)

                                    if self.fusion_type == 'add':
                                        entity_embedding = entity_embedding + company_subgraph_embedding
                                    # element-wise product
                                    elif self.fusion_type == 'ewp':
                                        entity_embedding_copy = entity_embedding.clone()
                                        entity_embedding = torch.mul(entity_embedding_copy, company_subgraph_embedding)
                                    elif self.fusion_type == 'att':
                                        entity_cat_subgraph = torch.cat([entity_embedding, company_subgraph_embedding]) # [dim * 2]
                                        #print(entity_cat_subgraph.size())
                                        #raise ValueError("For test.")
                                        att_weight = self.att_mlp(entity_cat_subgraph)
                                        entity_embedding = entity_embedding + company_subgraph_embedding * att_weight

                                  
                            # 判断节点与patent的连接关系
                            if (meta_type[3], meta_type[i]) in link_map:
                                connection_dict = link_map[(meta_type[3], meta_type[i])]
                                if entity_indices[i] in connection_dict[entity_indices[3]]:
                                    # 计算patent子图表征
                                    first_patentee_embeddings = self.first_patentee_embedding(torch.tensor(pid_first_patentee_connection[entity_indices[3]]).to(self.device))
                                    ipcClass_embeddings = self.ipcClass_embedding(torch.tensor(pid_ipcClass_connection[entity_indices[3]]).to(self.device))
                                    # 处理可能的字典缺失情况（针对测试集）
                                    if entity_indices[3] in patent_company_connection:
                                        company_embeddings = self.company_embedding(torch.tensor(patent_company_connection[entity_indices[3]]).to(self.device))
                                        attribute_embeddings = torch.cat([first_patentee_embeddings, ipcClass_embeddings, company_embeddings], dim=0)
                                    else:
                                        attribute_embeddings = torch.cat([first_patentee_embeddings, ipcClass_embeddings], dim=0)

                                    patent_embedding = self.patent_embedding(torch.tensor(entity_indices[3]).to(self.device))
                                    if self.aggregator == 'avg':
                                        patent_subgraph_embedding = patent_embedding + torch.mean(attribute_embeddings, dim=0)
                                    elif self.aggregator == 'ewp':
                                        info_sum = torch.zeros(self.embedding_dim).to(self.device)
                                        for t in range(attribute_embeddings.size(0)):
                                            attribute_emb = attribute_embeddings[t]
                                            info = torch.mul(attribute_emb, patent_embedding)
                                            info_sum = info_sum + info
                                        
                                        patent_subgraph_embedding = patent_embedding + info_sum

                                    elif self.aggregator == 'gate':
                                        # aggregate gate
                                        agg_representation = torch.zeros(self.embedding_dim).to(self.device)
                                        for t in range(attribute_embeddings.size(0)):
                                            attribute_emb = attribute_embeddings[t]
                                            pat_cat_attr = torch.cat([patent_embedding, attribute_emb]) # [dim * 2]
                                            agg_gate = self.agg_mlp(pat_cat_attr)
                                            agg_representation = agg_representation + torch.mul(attribute_emb, agg_gate)
                                        
                                        agg_representation = torch.div(agg_representation, attribute_embeddings.size(0))

                                        # filter gate
                                        avg_neighbor = torch.mean(attribute_embeddings, dim=0)
                                        pat_cat_avg_neighbor = torch.cat([patent_embedding, avg_neighbor])
                                        filter_gate = self.filter_mlp(pat_cat_avg_neighbor)
                                        all_one = torch.ones(self.embedding_dim).to(self.device)
                                        filter_representation = torch.mul(patent_embedding, (all_one - filter_gate))

                                        patent_subgraph_embedding = self.act(filter_representation + agg_representation)

                                    if self.fusion_type == 'add':
                                        entity_embedding = entity_embedding + patent_subgraph_embedding
                                    # element-wise product
                                    elif self.fusion_type == 'ewp':
                                        entity_embedding_copy = entity_embedding.clone()
                                        entity_embedding = torch.mul(entity_embedding_copy, patent_subgraph_embedding)
                                    elif self.fusion_type == 'att':
                                        entity_cat_subgraph = torch.cat([entity_embedding, patent_subgraph_embedding]) # [dim * 2]
                                        att_weight = self.att_mlp(entity_cat_subgraph)
                                        entity_embedding = entity_embedding + patent_subgraph_embedding * att_weight


                            entity_embeddings[i] = entity_embedding

                elif meta_type == "cplp":
                    entity_embeddings = torch.stack([
                        self.company_embedding(torch.tensor(entity_indices[0]).to(self.device)),
                        self.patent_embedding(torch.tensor(entity_indices[1]).to(self.device)),
                        self.ipcClass_embedding(torch.tensor(entity_indices[2]).to(self.device)),
                        self.patent_embedding(torch.tensor(entity_indices[3]).to(self.device))
                    ], dim=0)

                    if self.use_subgraph:
                        # 更新节点嵌入
                        for i in range(1, len(entity_embeddings) - 1):
                            entity_embedding = entity_embeddings[i]

                            # 判断节点与company的连接关系
                            if (meta_type[0], meta_type[i]) in link_map:
                                connection_dict = link_map[(meta_type[0], meta_type[i])]
                                if entity_indices[i] in connection_dict[entity_indices[0]]:
                                    # 计算company子图表征
                                    industry_embeddings = self.industry_embedding(torch.tensor(company_industry_connection[entity_indices[0]]).to(self.device))
                                    patent_embeddings = self.patent_embedding(torch.tensor(company_patent_connection[entity_indices[0]]).to(self.device))
                                    attribute_embeddings = torch.cat([industry_embeddings, patent_embeddings], dim=0)
                                    company_embedding = self.company_embedding(torch.tensor(entity_indices[0]).to(self.device))
                                    if self.aggregator == 'avg':
                                        company_subgraph_embedding = company_embedding + torch.mean(attribute_embeddings, dim=0)
                                    elif self.aggregator == 'ewp':
                                        info_sum = torch.zeros(self.embedding_dim).to(self.device)
                                        for t in range(attribute_embeddings.size(0)):
                                            attribute_emb = attribute_embeddings[t]
                                            info = torch.mul(attribute_emb, company_embedding)
                                            info_sum = info_sum + info
                                        
                                        company_subgraph_embedding = company_embedding + info_sum

                                    elif self.aggregator == 'gate':
                                        # aggregate gate
                                        agg_representation = torch.zeros(self.embedding_dim).to(self.device)
                                        for t in range(attribute_embeddings.size(0)):
                                            attribute_emb = attribute_embeddings[t]
                                            com_cat_attr = torch.cat([company_embedding, attribute_emb]) # [dim * 2]
                                            agg_gate = self.agg_mlp(com_cat_attr)
                                            agg_representation = agg_representation + torch.mul(attribute_emb, agg_gate)
                                        
                                        agg_representation = torch.div(agg_representation, attribute_embeddings.size(0))

                                        # filter gate
                                        avg_neighbor = torch.mean(attribute_embeddings, dim=0)
                                        com_cat_avg_neighbor = torch.cat([company_embedding, avg_neighbor])
                                        filter_gate = self.filter_mlp(com_cat_avg_neighbor)
                                        all_one = torch.ones(self.embedding_dim).to(self.device)
                                        filter_representation = torch.mul(company_embedding, (all_one - filter_gate))

                                        company_subgraph_embedding = self.act(filter_representation + agg_representation)

                                    if self.fusion_type == 'add':
                                        entity_embedding = entity_embedding + company_subgraph_embedding
                                    # element-wise product
                                    elif self.fusion_type == 'ewp':
                                        entity_embedding_copy = entity_embedding.clone()
                                        entity_embedding = torch.mul(entity_embedding_copy, company_subgraph_embedding)
                                    elif self.fusion_type == 'att':
                                        entity_cat_subgraph = torch.cat([entity_embedding, company_subgraph_embedding]) # [dim * 2]
                                        att_weight = self.att_mlp(entity_cat_subgraph)
                                        entity_embedding = entity_embedding + company_subgraph_embedding * att_weight


                            # 判断节点与patent的连接关系
                            if (meta_type[3], meta_type[i]) in link_map:
                                connection_dict = link_map[(meta_type[3], meta_type[i])]
                                if entity_indices[i] in connection_dict[entity_indices[3]]:
                                   # 计算patent子图表征
                                    first_patentee_embeddings = self.first_patentee_embedding(torch.tensor(pid_first_patentee_connection[entity_indices[3]]).to(self.device))
                                    ipcClass_embeddings = self.ipcClass_embedding(torch.tensor(pid_ipcClass_connection[entity_indices[3]]).to(self.device))
                                    # 处理可能的字典缺失情况
                                    if entity_indices[3] in patent_company_connection:
                                        company_embeddings = self.company_embedding(torch.tensor(patent_company_connection[entity_indices[3]]).to(self.device))
                                        attribute_embeddings = torch.cat([first_patentee_embeddings, ipcClass_embeddings, company_embeddings], dim=0)
                                    else:
                                        attribute_embeddings = torch.cat([first_patentee_embeddings, ipcClass_embeddings], dim=0)

                                    patent_embedding = self.patent_embedding(torch.tensor(entity_indices[3]).to(self.device))

                                    if self.aggregator == 'avg':
                                        patent_subgraph_embedding = patent_embedding + torch.mean(attribute_embeddings, dim=0)
                                    elif self.aggregator == 'ewp':
                                        info_sum = torch.zeros(self.embedding_dim).to(self.device)
                                        for t in range(attribute_embeddings.size(0)):
                                            attribute_emb = attribute_embeddings[t]
                                            info = torch.mul(attribute_emb, patent_embedding)
                                            info_sum = info_sum + info
                                        
                                        patent_subgraph_embedding = patent_embedding + info_sum

                                    elif self.aggregator == 'gate':
                                        # aggregate gate
                                        agg_representation = torch.zeros(self.embedding_dim).to(self.device)
                                        for t in range(attribute_embeddings.size(0)):
                                            attribute_emb = attribute_embeddings[t]
                                            pat_cat_attr = torch.cat([patent_embedding, attribute_emb]) # [dim * 2]
                                            agg_gate = self.agg_mlp(pat_cat_attr)
                                            agg_representation = agg_representation + torch.mul(attribute_emb, agg_gate)
                                        
                                        agg_representation = torch.div(agg_representation, attribute_embeddings.size(0))

                                        # filter gate
                                        avg_neighbor = torch.mean(attribute_embeddings, dim=0)
                                        pat_cat_avg_neighbor = torch.cat([patent_embedding, avg_neighbor])
                                        filter_gate = self.filter_mlp(pat_cat_avg_neighbor)
                                        all_one = torch.ones(self.embedding_dim).to(self.device)
                                        filter_representation = torch.mul(patent_embedding, (all_one - filter_gate))

                                        patent_subgraph_embedding = self.act(filter_representation + agg_representation)

                                    if self.fusion_type == 'add':
                                        entity_embedding = entity_embedding + patent_subgraph_embedding
                                    # element-wise product
                                    elif self.fusion_type == 'ewp':
                                        entity_embedding_copy = entity_embedding.clone()
                                        entity_embedding = torch.mul(entity_embedding_copy, patent_subgraph_embedding)
                                    elif self.fusion_type == 'att':
                                        entity_cat_subgraph = torch.cat([entity_embedding, patent_subgraph_embedding]) # [dim * 2]
                                        att_weight = self.att_mlp(entity_cat_subgraph)
                                        entity_embedding = entity_embedding + patent_subgraph_embedding * att_weight


                            entity_embeddings[i] = entity_embedding
                    
                else:
                    raise ValueError("Invalid meta_type. Must be 'cpfp' or 'cplp'.")

                path_embeddings.append(entity_embeddings)

        # 将嵌入序列输入到LSTM层
        path_embeddings = torch.stack(path_embeddings, dim=0)  # [num_paths, sequence_length, embedding_dim]
        # print(path_embeddings.size())
        # raise ValueError("For test.")
        _, hidden_state = self.lstm(path_embeddings)
        hidden_state = hidden_state[0].squeeze(0)  # [num_paths, hidden_dim]

        if self.merge_type == 'avg':
            # 求平均得到融合路径表征
            fused_path_representation = torch.mean(hidden_state, dim=0)  # [hidden_dim]
            
        elif self.merge_type == 'att':
            weights = []
            company_representation = path_embeddings[0, 0]
            for row_idx in range(hidden_state.size(0)):
                com_cat_path = torch.cat([company_representation, hidden_state[row_idx]]) # [self.embedding_dim + self.hidden_dim]
                weight = self.merge_mlp(com_cat_path)
                weights.append(weight)
            
            weights = torch.cat(weights) # [num_paths]
            weights_softmax = torch.nn.functional.softmax(weights, dim = 0)
            fused_path_representation = torch.sum(torch.mul(weights_softmax.unsqueeze(1).repeat(1, self.hidden_dim), hidden_state), dim = 0)  # [hidden_dim]

        fused_path_representations.append(fused_path_representation)
        
        # 将融合路径表征转换为张量
        fused_path_representations = torch.stack(fused_path_representations, dim=0)  # [batch_size, hidden_dim]

        # 通过MLP获得预测的(company, pid)匹配值
        prediction = self.mlp(fused_path_representations)  # [batch_size, 1]

        # 计算损失
        loss = self.bce_loss(prediction.cpu(), label)

        # 构建返回字典
        return_dict = {
            'loss': loss,
            'prediction': prediction
        }

        return return_dict

class PatentSubgraphPlus(nn.Module):
    def __init__(self, args, num_list, connection_list, device):
        super(PatentSubgraphPlus, self).__init__()
        self.embedding_dim = args.node_emb_size
        self.hidden_dim = args.hidden_emb_size
        self.num_list = num_list
        self.connection_list = connection_list
        self.device = device
        self.use_subgraph = args.use_subgraph
        self.aggregator = args.aggregator
        self.fusion_type = args.fusion_type
        self.merge_type = args.merge_type

        # 嵌入层
        self.patent_embedding = nn.Embedding(num_list[0], self.embedding_dim)
        self.patentee_embedding = nn.Embedding(num_list[1], self.embedding_dim)
        self.ipcSubClass_embedding = nn.Embedding(num_list[2], self.embedding_dim)
        self.industry_embedding = nn.Embedding(num_list[3], self.embedding_dim)
        self.appDate_embedding = nn.Embedding(num_list[4], self.embedding_dim)    

        # LSTM层
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)

        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # 融合子图表征时，注意力机制所用的MLP
        self.att_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

        # 获得子图表征时，采用gate形式时，aggregate gate所用的MLP
        self.agg_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Sigmoid()
        )

        # 获得子图表征时，采用gate形式时，filter gate所用的MLP
        self.filter_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Sigmoid()
        )

        # 获得子图表征时，采用gate形式时，最后所用的激活函数
        self.act = nn.LeakyReLU(negative_slope = 0.2)

        # 融合多路径表征时，采用的MLP
        self.merge_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim + self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # BCE损失函数
        self.bce_loss = nn.BCELoss()

        # 调用compute_subgraph_representation函数并将结果存储为模型内部属性
        #self.patent_subgraph_representation, self.company_subgraph_representation = self.compute_subgraph_representation()


    def forward(self, path_data_dict, label, link_map):
        # 定义一个列表存储融合路径表征
        fused_path_representations = []
        path_embeddings = []
        company_industry_connection = self.connection_list[0]
        company_patent_connection = self.connection_list[1]
        pid_first_patentee_connection = self.connection_list[2]
        pid_ipcSubClass_connection = self.connection_list[3]
        pid_appDate_connection = self.connection_list[4] 

        for meta_type, path_data in path_data_dict.items():
            c = path_data[0][0]
            p = path_data[0][3]
            break

        # 计算company子图表征 
        flag = 1
        if p in company_patent_connection[c]:
            p_list = company_patent_connection[c].copy()
            p_list.remove(p)
            if len(p_list) == 0:
                if c in company_industry_connection:
                    industry_embeddings = self.industry_embedding(torch.tensor(company_industry_connection[c]).to(self.device))                                            
                    attribute_embeddings = industry_embeddings # 二维
                else:
                    flag = 0
            else:
                patent_embeddings = self.patent_embedding(torch.tensor(p_list).to(self.device))
                if c in company_industry_connection:
                    industry_embeddings = self.industry_embedding(torch.tensor(company_industry_connection[c]).to(self.device))
                    attribute_embeddings = torch.cat([industry_embeddings, patent_embeddings], dim=0)
                else:
                    attribute_embeddings = patent_embeddings
        else:
            patent_embeddings = self.patent_embedding(torch.tensor(company_patent_connection[c]).to(self.device))
            if c in company_industry_connection:
                industry_embeddings = self.industry_embedding(torch.tensor(company_industry_connection[c]).to(self.device))
                attribute_embeddings = torch.cat([industry_embeddings, patent_embeddings], dim=0)
            else:
                attribute_embeddings = patent_embeddings

        company_embedding = self.patentee_embedding(torch.tensor(c).to(self.device)) # [dim]

        if flag == 0:
            company_subgraph_embedding = company_embedding
        elif self.aggregator == 'avg':
            company_subgraph_embedding = company_embedding + torch.mean(attribute_embeddings, dim=0)
        elif self.aggregator == 'ewp':
            info_sum = torch.zeros(self.embedding_dim).to(self.device)
            for t in range(attribute_embeddings.size(0)):
                attribute_emb = attribute_embeddings[t]
                info = torch.mul(attribute_emb, company_embedding)
                info_sum = info_sum + info
            
            company_subgraph_embedding = company_embedding + info_sum
        
        elif self.aggregator == 'gate':
            # aggregate gate
            agg_representation = torch.zeros(self.embedding_dim).to(self.device)
            for t in range(attribute_embeddings.size(0)):
                attribute_emb = attribute_embeddings[t] # [dim]
                com_cat_attr = torch.cat([company_embedding, attribute_emb]) # [dim * 2]
                agg_gate = self.agg_mlp(com_cat_attr)
                agg_representation = agg_representation + torch.mul(attribute_emb, agg_gate)
            
            agg_representation = torch.div(agg_representation, attribute_embeddings.size(0))

            # filter gate
            avg_neighbor = torch.mean(attribute_embeddings, dim=0)
            com_cat_avg_neighbor = torch.cat([company_embedding, avg_neighbor]) # [dim * 2]
            filter_gate = self.filter_mlp(com_cat_avg_neighbor)
            all_one = torch.ones(self.embedding_dim).to(self.device)
            filter_representation = torch.mul(company_embedding, (all_one - filter_gate))

            company_subgraph_embedding = self.act(filter_representation + agg_representation)

        # 计算patent子图表征
        first_patentee_embeddings = self.patentee_embedding(torch.tensor(pid_first_patentee_connection[p]).to(self.device))
        ipcSubClass_embeddings = self.ipcSubClass_embedding(torch.tensor(pid_ipcSubClass_connection[p]).to(self.device))
        appDate_embeddings = self.appDate_embedding(torch.tensor(pid_appDate_connection[p]).to(self.device))
        attribute_embeddings = torch.cat([first_patentee_embeddings, ipcSubClass_embeddings, appDate_embeddings], dim=0)
        patent_embedding = self.patent_embedding(torch.tensor(p).to(self.device))
        if self.aggregator == 'avg':
            patent_subgraph_embedding = patent_embedding + torch.mean(attribute_embeddings, dim=0)
        elif self.aggregator == 'ewp':
            info_sum = torch.zeros(self.embedding_dim).to(self.device)
            for t in range(attribute_embeddings.size(0)):
                attribute_emb = attribute_embeddings[t]
                info = torch.mul(attribute_emb, patent_embedding)
                info_sum = info_sum + info
            
            patent_subgraph_embedding = patent_embedding + info_sum

        elif self.aggregator == 'gate':
            # aggregate gate
            agg_representation = torch.zeros(self.embedding_dim).to(self.device)
            for t in range(attribute_embeddings.size(0)):
                attribute_emb = attribute_embeddings[t] # [dim]
                pat_cat_attr = torch.cat([patent_embedding, attribute_emb]) # [dim * 2]
                agg_gate = self.agg_mlp(pat_cat_attr)
                agg_representation = agg_representation + torch.mul(attribute_emb, agg_gate)
            
            agg_representation = torch.div(agg_representation, attribute_embeddings.size(0))

            # filter gate
            avg_neighbor = torch.mean(attribute_embeddings, dim=0)
            pat_cat_avg_neighbor = torch.cat([patent_embedding, avg_neighbor])
            filter_gate = self.filter_mlp(pat_cat_avg_neighbor)
            all_one = torch.ones(self.embedding_dim).to(self.device)
            filter_representation = torch.mul(patent_embedding, (all_one - filter_gate))

            patent_subgraph_embedding = self.act(filter_representation + agg_representation)


        # 处理路径集合
        for meta_type, path_data in path_data_dict.items():
            # 取部分路径，避免过拟合
            path_data = path_data[:5]
            # for case study
            if c == 814 and p == 1608:
                fw_results = open('result/patent_6600/results_23.txt', 'a+')
                line = meta_type + ':\n'
                fw_results.write(line)
                for pth in path_data:
                    pth_str = ','.join(map(str, pth))
                    line = pth_str + '\n'
                    fw_results.write(line)

                fw_results.close()                    

            # 将路径中的实体索引序列转换为嵌入序列
            for entity_indices in path_data:
                if meta_type == "cpfp":
                    entity_embeddings = torch.stack([
                        self.patentee_embedding(torch.tensor(entity_indices[0]).to(self.device)),
                        self.patent_embedding(torch.tensor(entity_indices[1]).to(self.device)),
                        self.patentee_embedding(torch.tensor(entity_indices[2]).to(self.device)),
                        self.patent_embedding(torch.tensor(entity_indices[3]).to(self.device))
                    ], dim=0)

                elif meta_type == "cpsp":
                    entity_embeddings = torch.stack([
                        self.patentee_embedding(torch.tensor(entity_indices[0]).to(self.device)),
                        self.patent_embedding(torch.tensor(entity_indices[1]).to(self.device)),
                        self.ipcSubClass_embedding(torch.tensor(entity_indices[2]).to(self.device)),
                        self.patent_embedding(torch.tensor(entity_indices[3]).to(self.device))
                    ], dim=0)

                elif meta_type == "cifp":
                    entity_embeddings = torch.stack([
                        self.patentee_embedding(torch.tensor(entity_indices[0]).to(self.device)),
                        self.industry_embedding(torch.tensor(entity_indices[1]).to(self.device)),
                        self.patentee_embedding(torch.tensor(entity_indices[2]).to(self.device)),
                        self.patent_embedding(torch.tensor(entity_indices[3]).to(self.device))
                    ], dim=0)

                else:
                    raise ValueError("Invalid meta_type. Must be 'cpfp', 'cpsp' or 'cifp'.")

                #print(entity_embeddings.size()) # torch.Size([4, 64])
                #raise ValueError("For test.")

                if self.use_subgraph:
                    # 更新节点嵌入
                    for i in range(1, len(entity_embeddings) - 1):
                        entity_embedding = entity_embeddings[i] # [dim]
                        # 判断节点与company的连接关系
                        if (meta_type[0], meta_type[i]) in link_map:
                            connection_dict = link_map[(meta_type[0], meta_type[i])]
                            if entity_indices[i] in connection_dict[entity_indices[0]]:

                                if self.fusion_type == 'add':
                                    entity_embedding = entity_embedding + company_subgraph_embedding
                                # element-wise product
                                elif self.fusion_type == 'ewp':
                                    entity_embedding_copy = entity_embedding.clone()
                                    entity_embedding = torch.mul(entity_embedding_copy, company_subgraph_embedding)
                                elif self.fusion_type == 'att':
                                    entity_cat_subgraph = torch.cat([entity_embedding, company_subgraph_embedding]) # [dim * 2]
                                    #print(entity_cat_subgraph.size())
                                    #raise ValueError("For test.")
                                    att_weight = self.att_mlp(entity_cat_subgraph)
                                    entity_embedding = entity_embedding + company_subgraph_embedding * att_weight
                            
                        # 判断节点与patent的连接关系
                        if (meta_type[3], meta_type[i]) in link_map:
                            connection_dict = link_map[(meta_type[3], meta_type[i])]
                            if entity_indices[i] in connection_dict[entity_indices[3]]:

                                if self.fusion_type == 'add':
                                    entity_embedding = entity_embedding + patent_subgraph_embedding
                                # element-wise product
                                elif self.fusion_type == 'ewp':
                                    entity_embedding_copy = entity_embedding.clone()
                                    entity_embedding = torch.mul(entity_embedding_copy, patent_subgraph_embedding)
                                elif self.fusion_type == 'att':
                                    entity_cat_subgraph = torch.cat([entity_embedding, patent_subgraph_embedding]) # [dim * 2]
                                    att_weight = self.att_mlp(entity_cat_subgraph)
                                    entity_embedding = entity_embedding + patent_subgraph_embedding * att_weight

                        entity_embeddings[i] = entity_embedding

                path_embeddings.append(entity_embeddings)

        # 将嵌入序列输入到LSTM层
        path_embeddings = torch.stack(path_embeddings, dim=0)  # [num_paths, sequence_length, embedding_dim]
        # print(path_embeddings.size())
        # raise ValueError("For test.")
        _, hidden_state = self.lstm(path_embeddings)
        hidden_state = hidden_state[0].squeeze(0)  # [num_paths, hidden_dim]

        if self.merge_type == 'avg':
            # 求平均得到融合路径表征
            fused_path_representation = torch.mean(hidden_state, dim=0)  # [hidden_dim]
            
        elif self.merge_type == 'att':
            weights = []
            company_representation = path_embeddings[0, 0]
            for row_idx in range(hidden_state.size(0)):
                com_cat_path = torch.cat([company_representation, hidden_state[row_idx]]) # [self.embedding_dim + self.hidden_dim]
                weight = self.merge_mlp(com_cat_path)
                weights.append(weight)
            
            weights = torch.cat(weights) # [num_paths]
            weights_softmax = torch.nn.functional.softmax(weights, dim = 0)
            # for case study
            if c == 814 and p == 1608:
                weights_softmax_copy = weights_softmax.detach().clone()
                weights_softmax_copy_cpu = weights_softmax_copy.cpu()
                weights_softmax_copy_cpu_list = weights_softmax_copy_cpu.tolist()
                weights_str = ','.join(map(str, weights_softmax_copy_cpu_list))                
                fw_results = open('result/patent_6600/results_23.txt', 'a+')
                line = 'weights_softmax:\n'
                fw_results.write(line)
                line = weights_str + '\n\n'
                fw_results.write(line)                    
                fw_results.close()

            fused_path_representation = torch.sum(torch.mul(weights_softmax.unsqueeze(1).repeat(1, self.hidden_dim), hidden_state), dim = 0)  # [hidden_dim]

        fused_path_representations.append(fused_path_representation)
        
        # 将融合路径表征转换为张量
        fused_path_representations = torch.stack(fused_path_representations, dim=0)  # [batch_size, hidden_dim]

        # 通过MLP获得预测的(company, pid)匹配值
        prediction = self.mlp(fused_path_representations)  # [batch_size, 1]

        # 计算损失
        loss = self.bce_loss(prediction.cpu(), label)

        # 构建返回字典
        return_dict = {
            'loss': loss,
            'prediction': prediction
        }

        return return_dict