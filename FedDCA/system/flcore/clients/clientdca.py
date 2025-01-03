import torch.nn as nn
import numpy as np
import time
from system.flcore.clients.clientbase import Client
import torch
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import copy
import torch.nn.functional as F

class clientDCA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )

        self.plocal_epochs = args.plocal_epochs
        self.global_model = copy.deepcopy(args.model)
        self.head_weights = [torch.ones_like(param.data).to(self.device)
                             for param in list(self.global_model.head.parameters())]
        self.eta = 1.0  # Weight learning rate. Default: 1.0
        self.lamda_mi = args.lamda_mi
        self.lamda_cl = args.lamda_cl

        self.mutual_info_weights = []
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for epoch in range(self.plocal_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.global_model.parameters()):
            old_param.data = new_param.data.clone()

        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    # -------------------------------------------------------------------------

    def adaptive_aggregate(self):
        trainloader = self.load_train_data()

        # 获取全局和本地头部模型参数的引用
        params_g = list(self.global_model.head.parameters())
        params = list(self.model.head.parameters())

        # 临时模型用于头部参数的学习
        model_t = copy.deepcopy(self.model)
        params_th = list(model_t.head.parameters())
        params_tb = list(model_t.base.parameters())

        # 临时模型用于头部参数的学习
        model_tg = copy.deepcopy(self.global_model)
        params_tgh = list(model_tg.head.parameters())
        params_tgb = list(model_tg.base.parameters())

        # 冻结底层参数以降低计算成本
        for param in params_tb:
            param.requires_grad = False
        for param_t in params_th:
            param_t.requires_grad = True

        # 冻结底层参数以降低计算成本
        for param_g in params_tgb:
            param_g.requires_grad = False
        for param_tg in params_tgh:
            param_tg.requires_grad = True

        # 使用SGD优化器，学习率设为0，仅用于计算梯度
        optimizer_t = torch.optim.SGD(model_t.parameters(), lr=0)
        optimizer_tg = torch.optim.SGD(model_tg.parameters(), lr=0)

        for epoch in range(self.plocal_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                optimizer_t.zero_grad()
                output_l = model_t(x)

                optimizer_tg.zero_grad()
                output_g = model_tg(x)

                # 分类损失
                loss_value = self.loss(output_l, y)

                # 增强互信息损失
                probs_local = F.softmax(output_l, dim=1)
                probs_global = F.softmax(output_g, dim=1)

                # 计算对比互信息 (CMI)
                entropy_local = -torch.mean(torch.sum(probs_local * torch.log(probs_local + 1e-8), dim=1))
                entropy_global = -torch.mean(torch.sum(probs_global * torch.log(probs_global + 1e-8), dim=1))

                mi_loss = entropy_local - entropy_global

                # 总损失（动态权重整合）
                total_loss = loss_value + 2 * mi_loss

                total_loss.backward()

                # Update head weights in this batch
                for param_t, param, param_g, weight in zip(params_th, params, params_g, self.head_weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1
                    )

                # Update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_th, params, params_g, self.head_weights):
                    param_t.data = param + (param_g - param) * weight

            # Obtain initialized aggregated head
            for param, param_t in zip(params, params_th):
                param.data = param_t.data.clone()


    # -------------------------------------------------------------------------


















    def adaptive_aggregate1(self):

        # 获取全局和本地 head 参数
        params_g = list(self.global_model.head.parameters())
        params = list(self.model.head.parameters())

        # 将本地 head 参数与全局 head 参数直接简单平均
        # 假设此处仅有一组 local model 和 global model 的聚合，如有多客户端，则需要在调用此函数前就将多个客户端参数收集后求平均
        for param_g, param in zip(params_g, params):
            # 简单平均：全局参数 = (全局参数 + 本地参数) / 2
            param_g.data = 0.5 * (param_g.data + param.data)

        # 将本地模型的 head 参数更新为聚合后的参数
        for param, param_g in zip(params, params_g):
            param.data = param_g.data.clone()

    def adaptive_aggregate1(self):
        trainloader = self.load_train_data()

        # aggregate head
        # obtain the references of the head parameters
        params_g = list(self.global_model.head.parameters())
        params = list(self.model.head.parameters())

        # temp local model only for head weights learning
        model_t = copy.deepcopy(self.model)
        params_th = list(model_t.head.parameters())
        params_tb = list(model_t.base.parameters())

        # frozen base to reduce computational cost in Pytorch
        for param in params_tb:
            param.requires_grad = False
        for param_t in params_th:
            param_t.requires_grad = True

        # used to obtain the gradient of model, no need to use optimizer.step(), so lr=0
        optimizer_t = torch.optim.SGD(model_t.parameters(), lr=0)

        # mutual_info_weights = []  # To store mutual information weights

        for epoch in range(self.plocal_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                optimizer_t.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y)

                # # Compute mutual information for current batch
                # probs = F.softmax(output, dim=1)
                # mi_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                # mutual_info_weights.append(mi_loss.item())

                loss_value.backward()

                # Update head weights in this batch
                for param_t, param, param_g, weight in zip(params_th, params, params_g, self.head_weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1
                    )

                # Update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_th, params, params_g, self.head_weights):
                    param_t.data = param + (param_g - param) * weight

        # # Normalize mutual information weights
        # mutual_info_weights = torch.tensor(mutual_info_weights)
        # normalized_weights = mutual_info_weights / mutual_info_weights.sum()

        # # Apply normalized weights to global head aggregation
        # for param_g, param, weight in zip(params_g, params, normalized_weights):
        #     param_g.data = weight * param.data + (1 - weight) * param_g.data

        # Obtain initialized aggregated head
        for param, param_t in zip(params, params_th):
            param.data = param_t.data.clone()

    def adaptive_aggregate1(self):
        trainloader = self.load_train_data()

        # aggregate head
        # obtain the references of the head parameters
        params_g = list(self.global_model.head.parameters())
        params = list(self.model.head.parameters())

        # temp local model only for head weights learning
        model_t = copy.deepcopy(self.model)
        params_th = list(model_t.head.parameters())
        params_tb = list(model_t.base.parameters())

        # frozen base to reduce computational cost in Pytorch
        for param in params_tb:
            param.requires_grad = False
        for param_t in params_th:
            param_t.requires_grad = True

        # used to obtain the gradient of model, no need to use optimizer.step(), so lr=0
        optimizer_t = torch.optim.SGD(model_t.parameters(), lr=0)

        for epoch in range(self.plocal_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                optimizer_t.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y)

                # Compute mutual information for current batch
                probs = F.softmax(output, dim=1)
                mi_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                self.mutual_info_weights.append(mi_loss.item())

                loss_value.backward()

                # Update head weights in this batch
                for param_t, param, param_g, weight in zip(params_th, params, params_g, self.head_weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1
                    )

                # Update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_th, params, params_g, self.head_weights):
                    param_t.data = param + (param_g - param) * weight

        # Normalize mutual information weights
        mutual_info_weights = torch.tensor(self.mutual_info_weights)
        normalized_weights = mutual_info_weights / mutual_info_weights.sum()

        # Apply normalized weights to global head aggregation
        for param_g, param, weight in zip(params_g, params, normalized_weights):
            param_g.data = weight * param.data + (1 - weight) * param_g.data

        # Obtain initialized aggregated head
        for param, param_t in zip(params, params_th):
            param.data = param_t.data.clone()


