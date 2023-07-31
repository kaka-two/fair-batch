import functools
from cmath import log
import math
import random
from re import sub
from typing import List
import queue
import env
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool

Batch_max = 50
container_type = 3
server_mode = False

# 比较函数
def compare(a, b):
    if a[0] != b[0]:
        return -1 if a[0] < b[0] else 1
    else:
        if a[1] < b[1]:
            return -1
        elif a[1] > b[1]:
            return 1
        else:
            return 0
# APGTO的辅助类  
class FS:
    def __init__(self, bandwidth_strategy:np.array, offloading_strategy:List[np.array], G_m:float):
        self.bandwidth_strategy = bandwidth_strategy.copy()
        self.offloading_strategy = [offloading_strategy[i].copy() for i in range(len(offloading_strategy))]
        self.G_m = G_m

# 主算法和对比算法      
class My_algo():
    def __init__(self, server: env.Server, environment: env.Environment, mobiles: List[env.Mobile],  loop_k=1, loop_l=1, server_num=3, container_num=7, time_slot=200, alpha_appro = 2, isTrue = 0) -> None:
        self.server = server
        self.environment = environment
        self.mobiles = mobiles      
        self.server_num = server_num
        self.container_num = container_num
        self.mobile_num = len(self.mobiles)
        self.loop_k = loop_k
        self.loop_l = loop_l
        self.time_slot = time_slot
        self.alpha_appro = alpha_appro
        self.isTrue = isTrue
        self.l_n_m = self.wire_trans_time()
        self.serve_rate = None
        self.serve_rate_1 = None
        self.serve_rate_2 = None
        self.serve_rate_3 = None
        if(isTrue == 0): self.left_GPU = [1 - self.server.pred_GPU[i] for i in range(time_slot)]
        else: self.left_GPU = [1 - self.server.true_GPU[i] for i in range(time_slot)]
     
    # 获取服务率
    def get_server_rate(self, t, offloading_strategy, bandwidth_strategy):
        total_com = 0
        com_num = np.zeros(container_type)
        total_num = np.zeros(container_type)
        server_rate = np.zeros(container_type + 1)
        # 统计服务率
        k_m = np.zeros(self.container_num, dtype=int)
        g_m = np.zeros(self.container_num)
        d_m = [10] * self.container_num
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])
                    k_m[j-1] += 1
        g_m = [self.server.inference_time[i][k_m[i] - 1] for i in range(self.container_num)]
        batch_time = np.zeros(self.container_num)
        for i in range(self.mobile_num):
            wireless_tran_rate = self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue]
            for j in range(self.container_num):
                if offloading_strategy[i][j+1]:
                    wireless_tran_time = self.environment.trans_time(wireless_tran_rate, self.server.input_size[j])
                    wire_tran_time = self.l_n_m[i][j]
                    batch_time[j] = np.max([batch_time[j], wireless_tran_time + wire_tran_time])
        com = [(batch_time[i] + g_m[i]) for i in range(self.container_num)]

        for i in range(self.mobile_num):
            total_num[self.mobiles[i].task[t][0]] += 1
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    if(com[j-1] <= self.mobiles[i].task[t][2]):
                        com_num[self.mobiles[i].task[t][0]] += 1
                        total_com += 1
        server_rate[0] = total_com / self.mobile_num
        for i in range(len(server_rate)-1):
            server_rate[i+1] = com_num[i] / total_num[i]
        return server_rate

    # 检查卸载约束
    def cheak_offload(self, offloading_strategy):
        for i in range(len(offloading_strategy)):
            offloaded = 0
            for j in range(len(offloading_strategy[0])):
                if(offloading_strategy[i][j] == 1):  offloaded += 1
            if(offloaded != 1): return False
        return True
    
    # 检查容量约束
    def check_capacity(self, t, offloading_strategy):
        k_m = np.zeros(self.container_num, dtype=int)
        k_max = [np.floor(self.server.container_capacity[i] * (1 - self.server.true_GPU[t]) / self.server.workload[i]) for i in range(self.container_num)]
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    k_m[j-1] += 1
        for i in range(1, len(offloading_strategy[0])):
            if(k_m[i-1] > k_max[i-1]):   return 0
            
        return 1

    # 使结果满足deadline约束
    def deadline_constaints(self, t, bandwidth_strategy, offloading_strategy, sigma, time_cost) -> np.array:
        # 清楚不符合约束的卸载
        k_max = [self.server.container_capacity[i] * (1 - self.server.true_GPU[t]) / self.server.workload[i] for i in range(self.container_num)]
        k_m = np.zeros(self.container_num, dtype=int)
        g_m = np.zeros(self.container_num)
        d_m = [10] * self.container_num
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])
                    k_m[j-1] += 1
        g_m = [self.server.inference_time[i][k_m[i] - 1] for i in range(self.container_num)]
        batch_time = np.zeros(self.container_num)
        for i in range(self.mobile_num):
            wireless_tran_rate = self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue]
            for j in range(self.container_num):
                if offloading_strategy[i][j+1]:
                    wireless_tran_time = self.environment.trans_time(wireless_tran_rate, self.server.input_size[j])
                    wire_tran_time = self.l_n_m[i][j]
                    batch_time[j] = np.max([batch_time[j], wireless_tran_time + wire_tran_time])
        com = [(batch_time[i] + g_m[i]) for i in range(self.container_num)]

        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    if(com[j-1] + time_cost > self.mobiles[i].task[t][2] ):
                        offloading_strategy[i][j] = 0
        # 重新计算tau
        k_m = np.zeros(self.container_num, dtype=int)
        g_m = np.zeros(self.container_num)
        d_m = [10] * self.container_num
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])
                    k_m[j-1] += 1
        g_m = [self.server.inference_time[i][k_m[i] - 1] for i in range(self.container_num)]

        batch_time = np.zeros(self.container_num)
        for i in range(self.mobile_num):
            wireless_tran_rate = self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue]
            for j in range(self.container_num):
                if offloading_strategy[i][j+1]:
                    wireless_tran_time = self.environment.trans_time(wireless_tran_rate, self.server.input_size[j])
                    wire_tran_time = self.l_n_m[i][j]
                    batch_time[j] = np.max([batch_time[j], wireless_tran_time + wire_tran_time])
        tau = [(batch_time[i] + g_m[i]) / d_m[i] for i in range(self.container_num)]

        # 贪心卸载
        r_n = [self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue] for i in range(self.mobile_num)]   
        batching_time = batch_time
        batch_size = k_m
        for i in range(self.mobile_num):
            offed = False
            for j in range(self.container_num + 1):
                if(offloading_strategy[i][j] == 1): offed = True
            if(offed):   continue

            off_server = 1
            min_time = 500.0
            for j in range(self.container_num + 1):
                if(j == 0): continue
                batching_time_j = np.max([batching_time[j - 1], self.server.input_size[j - 1] / r_n[i] + self.l_n_m[i][j - 1]])
                batch_size_j = batch_size[j - 1] + 1
                d_m_j = np.min([d_m[j - 1], self.mobiles[i].task[t][2]])
                if(sigma[t][i][j] == 1 and (batching_time_j + self.server.inference_time[j - 1][batch_size_j]) / d_m_j < min_time and batch_size[j-1] < k_max[j-1]):
                    off_server = j
                    temp_batching_time = np.max([batching_time[j - 1], self.server.input_size[j - 1] / r_n[i] + self.l_n_m[i][j - 1]])
                    temp_batch_size = batch_size[j - 1] + 1
                    temp_d_m_j = np.min([d_m[j - 1], self.mobiles[i].task[t][2]])
                    min_time =  (batching_time_j + self.server.inference_time[j - 1][batch_size_j]) / d_m_j
            offloading_strategy[i][off_server] = 1
            batching_time[off_server - 1] = temp_batching_time
            batch_size[off_server - 1] = temp_batch_size
            d_m[off_server - 1] = temp_d_m_j
        return offloading_strategy                
    
    # 使结果满足容量约束
    def capacity_constaints(self, t, bandwidth_strategy, offloading_strategy, sigma) -> np.array:
        # 清楚不符合约束的卸载
        k_m = np.zeros(self.container_num, dtype=int)
        k_max = [self.server.container_capacity[i] * (1 - self.server.true_GPU[t]) / self.server.workload[i] for i in range(self.container_num)]
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    k_m[j-1] += 1
                    if(k_m[j-1] > k_max[j-1]):
                        offloading_strategy[i][j] = 0
        # 重新计算tau
        k_m = np.zeros(self.container_num, dtype=int)
        g_m = np.zeros(self.container_num)
        d_m = [10] * self.container_num
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])
                    k_m[j-1] += 1
        g_m = [self.server.inference_time[i][k_m[i] - 1] for i in range(self.container_num)]

        batch_time = np.zeros(self.container_num)
        for i in range(self.mobile_num):
            wireless_tran_rate = self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue]
            for j in range(self.container_num):
                if offloading_strategy[i][j+1]:
                    wireless_tran_time = self.environment.trans_time(wireless_tran_rate, self.server.input_size[j])
                    wire_tran_time = self.l_n_m[i][j]
                    batch_time[j] = np.max([batch_time[j], wireless_tran_time + wire_tran_time])
        tau = [(batch_time[i] + g_m[i]) / d_m[i] for i in range(self.container_num)]
    
        # 贪心卸载
        r_n = [self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue] for i in range(self.mobile_num)]   
        batch_size = k_m
        batching_time = batch_time
        tau_m = tau
        for i in range(self.mobile_num):
            offed = False
            for j in range(self.container_num + 1):
                if(offloading_strategy[i][j] == 1): offed = True
            if(offed):   continue

            off_server = 1
            min_tau = 500.0
            for j in range(self.container_num + 1):
                if(sigma[t][i][j] == 1 and tau_m[j - 1] < min_tau and batch_size[j-1] < k_max[j-1]):
                    off_server = j
                    temp_batching_time = np.max([batching_time[j - 1], self.server.input_size[j - 1] / r_n[i] + self.l_n_m[i][j - 1]])
                    temp_batch_size = int(batch_size[j - 1] + 1)
                    temp_d_m_j = np.min([d_m[j - 1], self.mobiles[i].task[t][2]])
                    min_tau =  tau_m[j - 1]
            
            offloading_strategy[i][off_server] = 1
            if(off_server != 0):
                batching_time[off_server - 1] = temp_batching_time
                batch_size[off_server - 1] = temp_batch_size
                d_m[off_server - 1] = temp_d_m_j
                tau_m[j - 1] = (temp_batching_time + self.server.inference_time[off_server - 1][temp_batch_size]) / temp_d_m_j
        return offloading_strategy

    #########################功能函数###################
    # 初始的带宽分配策略——均分
    def Average_allocate_bandwidth(self, server_bandwidth: np.array) -> np.array:
        connect_server_num = np.zeros(self.server_num)
        ave_bandwidth_strategy = np.zeros([self.mobile_num])
        for mobile in self.mobiles:
            connect_server_num[mobile.connected_server] += 1
        for i in range(self.mobile_num):
            ave_bandwidth_strategy[i] = server_bandwidth[self.mobiles[i].connected_server] / connect_server_num[self.mobiles[i].connected_server]
        return ave_bandwidth_strategy

    # 获取满足任务精度，类型的矩阵sigma以及每个服务器可能卸载任务的最小deadline
    def gen_sigma(self) -> tuple([np.array, np.array]):
        sigma = []
        d_m_min = []
        d_m_in_this_slot  = [10] * self.container_num 
        sigma_in_this_slot = np.zeros([self.mobile_num, self.container_num + 1])
        for t in range(self.time_slot):
            mobile_index = 0
            for mobile in self.mobiles:
                if self.server.container_type[mobile.deployed_container] == mobile.task[t][0] and mobile.task[t][1] >= self.server.accuracy[mobile.deployed_container]:
                    sigma_in_this_slot[mobile_index][0] = 1
                for s in range(self.container_num):
                    if self.server.container_type[s] == mobile.task[t][0] and mobile.task[t][1] >= self.server.accuracy[s]:
                        sigma_in_this_slot[mobile_index][s + 1] = 1
                        d_m_in_this_slot[s] = np.min([d_m_in_this_slot[s], mobile.task[t][2]]) 
                mobile_index += 1
            sigma.append(sigma_in_this_slot)
            d_m_min.append(d_m_in_this_slot)
            sigma_in_this_slot = np.zeros([self.mobile_num, self.container_num + 1])
            d_m_in_this_slot = [10] * self.container_num 
        return sigma, d_m_min

    # 获取有线传输时间
    def wire_trans_time(self) -> np.array:
        l_n_m = [[10] * self.container_num] * self.mobile_num
        for i in range(self.mobile_num):
            for j in range(self.container_num):
                if not j in self.server.deployed_container[self.mobiles[i].connected_server]:
                    l_n_m[i][j] = self.environment.trans_time(self.environment.fibic_network, self.server.input_size[j]) # 单位：s
        return l_n_m

    # 通过传输速率求带宽
    def cal_bandwidth_for_one(self, r_n, i, t) -> float:
        p_n = self.mobiles[0].trans_power
        g_n = self.environment.channel_gain(self.mobiles[i].distance, t)[self.isTrue]
        w_o = self.environment.noise
        b_n = 8 * r_n / math.log2(1 + p_n*g_n/w_o)
        return b_n
 
    def cal_bandwidth(self, r_n, t) -> np.array:
        p_n = self.mobiles[0].trans_power
        g_n = [self.environment.channel_gain(self.mobiles[i].distance, t)[self.isTrue] for i in range(self.mobile_num)]
        w_o = self.environment.noise
        b_n = [8 * r_n[i] / math.log2(1 + p_n*g_n[i]/w_o) for i in range(self.mobile_num)]
        return b_n

    # 按斜对角拼接矩阵
    def adjConcat(self, a: np.array, b: np.array):
        if len(b) == 0 or len(b[0]) == 0: return a
        rowa = len(a)
        cola = len(a[0])
        rowb = len(b)
        colb = len(b[0])
       
        left = np.append(a, np.zeros([rowb, cola]), axis = 0)
        right = np.append(np.zeros([rowa, colb]), b, axis = 0)
        result = np.append(left, right, axis = 1)
        return result

    # 根据真实值求tau
    def cal_tau_for_true(self, t, bandwidth_strategy, offloading_strategy, cost_in_this_slot, is_for_BFTUA, sigma) -> float:
        offloading_strategy = offloading_strategy
        bandwidth_strategy = bandwidth_strategy
        tau = 0
        k_m = np.zeros(self.container_num, dtype=int)
        g_m = np.zeros(self.container_num)
        d_m = [10] * self.container_num
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if is_for_BFTUA:
                    if sigma[t][i][j] == 1:
                        d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])
                else:
                    if offloading_strategy[i][j] == 1:
                        d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])
                if offloading_strategy[i][j] == 1:
                    k_m[j-1] += 1
        g_m = [self.server.inference_time[i][k_m[i] - 1] for i in range(self.container_num)]
        # 获取成批时间
        batch_time = np.zeros(self.container_num)
        for i in range(self.mobile_num):
            wireless_tran_rate = self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue]
            for j in range(self.container_num):
                if offloading_strategy[i][j+1]:
                    wireless_tran_time = self.environment.trans_time(wireless_tran_rate, self.server.input_size[j])
                    wire_tran_time = self.l_n_m[i][j]
                    batch_time[j] = np.max([batch_time[j], wireless_tran_time + wire_tran_time])
        # 卸载到本地
        for i in range(len(offloading_strategy)):
            if offloading_strategy[i][0] == 1:
                deployed_container = self.mobiles[i].deployed_container
                local_infer_time = self.mobiles[0].local_inference_time[deployed_container]
                task_deadline = self.mobiles[i].task[t][2]
                tau = np.max([tau, (local_infer_time + cost_in_this_slot) / task_deadline])
        # 卸载到服务器
        for i in range(self.container_num):
            tau = np.max([tau, (batch_time[i] + g_m[i] + cost_in_this_slot) / d_m[i]])
        return tau

    # 根据卸载策略和带宽分配策略求出tau
    def cal_tau(self, t, bandwidth_strategy, offloading_strategy, cost_in_this_slot, is_for_BFTUA, sigma) -> float:
        offloading_strategy = offloading_strategy
        bandwidth_strategy = bandwidth_strategy
        tau = 0
        k_m = np.zeros(self.container_num, dtype=int)
        g_m = np.zeros(self.container_num)
        d_m = [10] * self.container_num
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if is_for_BFTUA:
                    if sigma[t][i][j] == 1:
                        d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])
                else:
                    if offloading_strategy[i][j] == 1:
                        d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])
                if offloading_strategy[i][j] == 1:
                    k_m[j-1] += 1
        g_m = [self.server.inference_time[i][k_m[i] - 1] for i in range(self.container_num)]
        # 获取成批时间
        batch_time = np.zeros(self.container_num)
        for i in range(self.mobile_num):
            wireless_tran_rate = self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue]
            for j in range(self.container_num):
                if offloading_strategy[i][j+1]:
                    wireless_tran_time = self.environment.trans_time(wireless_tran_rate, self.server.input_size[j])
                    wire_tran_time = self.l_n_m[i][j]
                    batch_time[j] = np.max([batch_time[j], wireless_tran_time + wire_tran_time])

        # 卸载到本地
        for i in range(len(offloading_strategy)):
            if offloading_strategy[i][0] == 1:
                deployed_container = self.mobiles[i].deployed_container
                local_infer_time = self.mobiles[0].local_inference_time[deployed_container]
                task_deadline = self.mobiles[i].task[t][2]
                tau = np.max([tau, (local_infer_time + cost_in_this_slot) / task_deadline])
        # 卸载到服务器
        for i in range(self.container_num):
            tau = np.max([tau, (batch_time[i] + g_m[i] + cost_in_this_slot) / d_m[i]])
        return tau
    
     #########################主算法函数###################
    # 已知带宽策略求卸载策略
    def BFTUA(self, t, bandwidth_strategy, sigma, alpha_appro, tau_max, process_in_local) -> tuple([np.array, float]):
        epsilon = 0.05
        offloading_strategy = None
        r_n = [self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue] for i in range(self.mobile_num)]
        tau_min = 10
        d_m = [10] * self.container_num
        for i in range(self.mobile_num):
            for j in range(self.container_num):
                if sigma[t][i][j+1] == 1:
                    d_m[j] = np.min([self.mobiles[i].task[t][2], d_m[j]]) 
        d_min = np.min(d_m)
        for i in range(self.mobile_num):
            if r_n[i] * d_min * alpha_appro > 0:
                tau_min = np.min([tau_min, np.min(self.server.input_size) * 2 / (r_n[i] * d_min * alpha_appro)])
        tau_max = np.max([tau_max,tau_min]) + epsilon
        tau_mid = (tau_min + tau_max) / 2
        while tau_max - tau_min > epsilon:
            feasible, x_tmp =  self.FEASIBLE(t, r_n, sigma, alpha_appro, tau_mid, d_m, process_in_local)
            if feasible:
                offloading_strategy = x_tmp
                tau_max = tau_mid
            else:
                tau_min = tau_mid
            tau_mid = (tau_min + tau_max) / 2
        #-------------取得FBAA的tau搜索上界----------------#
        if offloading_strategy is None: 
            _, x_tmp =  self.FEASIBLE(t, r_n, sigma, alpha_appro, 0.95, d_m, process_in_local)
            return x_tmp, 3
        tau_max = self.cal_tau(t, bandwidth_strategy, offloading_strategy, 0, False, sigma)
        return offloading_strategy, tau_max
   
    # 判断tau是否存在alpha适应性的卸载方案
    def FEASIBLE(self, t, r_n, sigma, alpha_appro, tau_mid, d_m,  process_in_local) -> tuple([bool, List[np.array]]):
        k_max = np.zeros(self.container_num)
        g_max = [alpha_appro * tau_mid * d_m[i] / 2 for i in range(self.container_num)]
        for i in range(self.container_num):
            for j in range(np.min([self.mobile_num, Batch_max])):
                if self.server.inference_time[i][j] > g_max[i]:
                    break
                k_max[i] = j
        k_max = [np.min([k_max[i], (self.server.container_capacity[i] + self.server.GPU_gap * self.left_GPU[t]) / self.server.workload[i]]) for i in range(self.container_num)]
        
        k_max = np.floor(k_max)

        # 构建残差网络
        theta = np.zeros([self.mobile_num, self.container_num + 1])
        n_m = np.zeros([self.mobile_num, self.container_num + 1], dtype=int)
        for i in range(self.mobile_num):
            theta[i][0] = alpha_appro * tau_mid * self.mobiles[i].task[t][2] / (2 * self.mobiles[i].local_inference_time[self.mobiles[i].deployed_container])
            if theta[i][0] >= 1 and sigma[t][i][0] >= 1:
                n_m[i][0] = 1
            for j in range(self.container_num):
                theta[i][j+1] = alpha_appro * tau_mid * r_n[i] * d_m[j] / (2 * (self.server.input_size[j] + r_n[i] * self.l_n_m[i][j]))
                if theta[i][j+1] >= 1 and sigma[t][i][j+1] >= 1:
                    n_m[i][j+1] = 1

        n = 2 + self.mobile_num + self.container_num + 1
        s_n = np.ones([1, self.mobile_num], dtype=int)
        m_t = np.zeros([self.container_num + 1, 1], dtype=int)
        for i in range(self.container_num):
            m_t[i+1][0] = k_max[i]
        m_t[0][0] = self.mobile_num if process_in_local else 0

        x_n_m = np.zeros([self.mobile_num, self.container_num + 1])
        x_num = 0
        sub_x_nums = np.zeros(self.server.type_num)
        tmap, smap = self.mapType(t)

        for i in range(self.server.type_num):
            self.subFlow(tmap, smap, n_m, x_n_m, sub_x_nums, i, k_max, process_in_local)
        
        x_num = sub_x_nums.sum()
        if(x_num >= self.mobile_num):         
            return True, x_n_m
        else:
            return False, x_n_m

    def subFlow(self, tmap, smap, n_m, x_n_m, sub_x_nums, i, k_max, process_in_local):
        sub_n = 2 + len(tmap[i]) + len(smap[i]) + 1
        sub_s_n = np.ones([1, len(tmap[i])], dtype=int)
        sub_n_m = np.zeros([len(tmap[i]), len(smap[i]) + 1])
        sub_m_t = np.zeros([len(smap[i]) + 1, 1], dtype=int)
        for j in range(len(sub_n_m)):
            for k in range(len(sub_n_m[0])):
                if(k == 0):
                    if(n_m[tmap[i][j]][0] == 1): sub_n_m[j][0] = 1
                else:
                    if(n_m[tmap[i][j]][smap[i][k - 1] + 1] == 1):   sub_n_m[j][k] = 1
        for j in range(len(smap[i])):
            sub_m_t[j+1][0] = k_max[smap[i][j]]
        sub_m_t[0][0] = len(tmap[i]) if process_in_local else 0
        res = self.adjConcat(sub_s_n, sub_n_m)
        res = self.adjConcat(res, sub_m_t)
        res = np.concatenate((np.zeros([sub_n-1, 1], dtype=int), res), axis = 1)
        res = np.concatenate((res, np.zeros([1, sub_n], dtype=int)), axis = 0)
        maxflow = Maxflow(res = res, mobile_num = len(tmap[i]), container_num = len(smap[i]) + 1)
        sub_x_num, sub_x_n_m = maxflow.dicnic()

        sub_x_nums[i] = sub_x_num
        for j in range(len(sub_x_n_m)):
            for k in range(len(sub_x_n_m[0])):
                if(sub_x_n_m[j][k] == 1):
                    if(k == 0):
                        x_n_m[tmap[i][j]][0] = 1
                    else:
                        x_n_m[tmap[i][j]][smap[i][k - 1] + 1] = 1

    def mapType(self, t):
        tmap = []
        smap = []
        for i in range(self.server.type_num):
            tdic = dict()
            sdic = dict()
            l = 0
            m = 0
            for j in range(self.mobile_num):
                if(self.mobiles[j].task[t][0] == i):
                    tdic[l] = j
                    l += 1
            for j in range(self.container_num):
                if(self.server.container_type[j] == i):
                    sdic[m] = j
                    m += 1
            tmap.append(tdic)
            smap.append(sdic)
        return tmap, smap            

    # 已知卸载策略求带宽策略
    def FBAA(self, t, offloading_strategy, tau_max, cost_in_this_slot, sigma) -> tuple([np.array, float, float]):
        bandwidth_strategy = np.zeros([self.mobile_num])
        epsilon = 0.001
        k_m = np.zeros(self.container_num, dtype=int)
        d_m = [10] * self.container_num
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    k_m[j-1] += 1
                    d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])         
        g_m =[self.server.inference_time[i][k_m[i]-1] for i in range(self.container_num)]
        beta_m = [g_m[i] / d_m[i] for i in range(self.container_num)]
        tau_min = np.max(beta_m) - epsilon
        tau_max = 3
        tau_mid = (tau_min + tau_max) / 2
        while tau_max - tau_min > epsilon:
            feasible = True
            r_n = np.zeros(self.mobile_num)
            for i in range(self.mobile_num):
                for j in range(self.container_num):
                    if  offloading_strategy[i][j+1]:
                        a = self.server.input_size[j]
                        b = d_m[j] * (tau_mid - beta_m[j])
                        c = self.l_n_m[i][j]
                        if b - c > 0:
                            r_n[i] = r_n[i] + a / (b - c)
                        else:
                            feasible = False
            tmp_bandwidth_strategy = self.cal_bandwidth(r_n, t)
            total_B = np.zeros(self.server_num)
            for i in range(self.mobile_num):
                total_B[self.mobiles[i].connected_server] += tmp_bandwidth_strategy[i]
            for i in range(self.server_num):
                if total_B[i] > self.server.total_bandwidth[i]:
                    feasible = False  
            if feasible:
                tau_max = tau_mid
                bandwidth_strategy = tmp_bandwidth_strategy
            else:
                tau_min = tau_mid
            tau_mid = (tau_min + tau_max) / 2
        total_B = np.zeros(self.server_num)
        for i in range(self.mobile_num):
            total_B[self.mobiles[i].connected_server] += bandwidth_strategy[i]
        remain_bandwidth = [self.server.total_bandwidth[i] - total_B[i] for i in range(self.server_num)]
        # remain_bandwidth_rate = [(self.server.total_bandwidth[i] - total_B[i]) /  self.server.total_bandwidth[i] for i in range(self.server_num)]
        remain_bandwidth_strategy = self.Average_allocate_bandwidth(remain_bandwidth)
        bandwidth_strategy = [bandwidth_strategy[i] + remain_bandwidth_strategy[i] for i in range(self.mobile_num)]
        # print("remain_rate")
        # print(remain_bandwidth_rate)
        #-------------取得BFTUA的tau搜索上界tau_max以及精确tmp_tau----------------#
        tmp_tau = self.cal_tau(t, bandwidth_strategy, offloading_strategy, cost_in_this_slot, False, sigma)
        tau_max = self.cal_tau(t, bandwidth_strategy, offloading_strategy, 0, True, sigma)
        
        return bandwidth_strategy, tmp_tau, tau_max

    # 资源紧缺下的贪婪的带宽分配策略
    def FBAAForOut(self, t, offloading_strategy, tau_max, cost_in_this_slot, sigma):
        bandwidth_strategy = np.zeros(self.mobile_num)
        need_band = np.zeros([self.mobile_num, 2])
        for i in range(self.mobile_num):
            need_band[i][1] = i
        epsilon = 0.001
        k_m = np.zeros(self.container_num, dtype=int)
        d_m = [10] * self.container_num
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    k_m[j-1] += 1
                    d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])
        g_m =[self.server.inference_time[i][k_m[i]-1] for i in range(self.container_num)]
        beta_m = [g_m[i] / d_m[i] for i in range(self.container_num)]
        r_n = np.zeros(self.mobile_num)
        total_B = self.server.total_bandwidth.copy()
        for i in range(self.mobile_num):
            for j in range(self.container_num):
                if  offloading_strategy[i][j+1] == 1:
                    a = self.server.input_size[j]
                    b = d_m[j] * (0.95 - beta_m[j])
                    c = self.l_n_m[i][j]
                    if b - c > 0:
                        r_n[i] = r_n[i] + a / (b - c)
                    need_band[i][0] = self.cal_bandwidth_for_one(r_n[i], i, t)

        need_band = sorted(need_band, key=functools.cmp_to_key(compare))
        com = 0
        for i in range(self.mobile_num):
            m = int(need_band[i][1])
            b = need_band[i][0]
            if (total_B[self.mobiles[m].connected_server] > b): 
                com += 1
                bandwidth_strategy[m] = np.min([b, total_B[self.mobiles[m].connected_server]])
                total_B[self.mobiles[m].connected_server] -= bandwidth_strategy[m]
            else:
                offloading_strategy[m][0] = 0
                for j in range(self.container_num):
                    offloading_strategy[m][j] = 0
        remain_bandwidth_strategy = self.Average_allocate_bandwidth(total_B)
        bandwidth_strategy = [bandwidth_strategy[i] + remain_bandwidth_strategy[i] for i in range(self.mobile_num)]
        
        #-------------取得BFTUA的tau搜索上界tau_max以及精确tmp_tau----------------#
        tmp_tau = self.cal_tau(t, bandwidth_strategy, offloading_strategy, cost_in_this_slot, False, sigma)
        tau_max = self.cal_tau(t, bandwidth_strategy, offloading_strategy, 0, True, sigma)
        
        return bandwidth_strategy, offloading_strategy

    
    # 贪婪的卸载策略
    def greedyO(self, t, bandwidth_strategy, sigma, alpha_appro, tau_max, process_in_local)  -> tuple([np.array, float]):
        offloading_strategy = np.zeros([self.mobile_num, self.container_num + 1])
        r_n = [self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue] for i in range(self.mobile_num)]   
        batching_time = np.zeros(self.container_num)
        batch_size = np.zeros(self.container_num)
        d_m = np.ones(self.container_num)
        tau_m = np.ones(self.container_num) * 0.5
        for i in range(self.mobile_num):
            off_server = 1
            min_tau = 500.0
            for j in range(self.container_num + 1):
                if(j == 0):
                    if(not process_in_local):
                        continue
                    if(sigma[t][i][0] == 1 and self.mobiles[i].local_inference_time[self.mobiles[i].deployed_container] / self.mobiles[i].task[t][2] < min_tau):
                        off_server = j
                        min_tau = self.mobiles[i].local_inference_time[self.mobiles[i].deployed_container] / self.mobiles[i].task[t][2]
                        continue
                if(sigma[t][i][j] == 1 and tau_m[j - 1] < min_tau):
                    off_server = j
                    temp_batching_time = np.max([batching_time[j - 1], self.server.input_size[j - 1] / r_n[i] + self.l_n_m[i][j - 1]])
                    temp_batch_size = int(batch_size[j - 1] + 1)
                    temp_d_m_j = np.min([d_m[j - 1], self.mobiles[i].task[t][2]])
                    min_tau =  tau_m[j - 1]
            
            offloading_strategy[i][off_server] = 1
            if(off_server != 0):
                batching_time[off_server - 1] = temp_batching_time
                batch_size[off_server - 1] = temp_batch_size
                d_m[off_server - 1] = temp_d_m_j
                tau_m[j - 1] = (temp_batching_time + self.server.inference_time[off_server - 1][temp_batch_size]) / temp_d_m_j
        #-------------取得FBAA的tau搜索上界----------------#
        if offloading_strategy is None: return None, 3
        tau_max = self.cal_tau(t, bandwidth_strategy, offloading_strategy, 0, False, sigma)
        return offloading_strategy, tau_max

        

    #########################简单的对比算法函数###################
    # 平均分配的带宽分配策略
    def AVEB(self, t, offloading_strategy, tau_max, cost_in_this_slot, sigma) -> tuple([np.array, float, float]):
        connect_mobile_num = np.zeros(self.server_num)
        bandwidth_strategy = np.zeros([self.mobile_num])
        for i in range(self.mobile_num):
            if offloading_strategy[i][0] == 0:
                connect_mobile_num[self.mobiles[i].connected_server] += 1
        for i in range(self.mobile_num):
            if offloading_strategy[i][0] == 0:
                bandwidth_strategy[i] = self.server.total_bandwidth[self.mobiles[i].connected_server] / connect_mobile_num[self.mobiles[i].connected_server]
        
        #-------------取得BFTUA的tau搜索上界tau_max以及精确tmp_tau----------------#
        tmp_tau = self.cal_tau(t, bandwidth_strategy, offloading_strategy, cost_in_this_slot, False, sigma)
        tau_max = self.cal_tau(t, bandwidth_strategy, offloading_strategy, 0, True, sigma)
        return bandwidth_strategy, tmp_tau, tau_max

    # 就近的卸载策略
    def Nearest(self, t, bandwidth_strategy, sigma, tau_max) -> tuple([np.array, float]):
        r_n = [self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue] for i in range(self.mobile_num)]
        offloading_strategy = np.zeros([self.mobile_num, self.container_num+1])
        # print(sigma[t])
        for i in range(self.mobile_num):
            is_offloaded = False
            # # 查询本地
            # if sigma[t][i][0] == 1 and self.mobiles[i].local_inference_time[self.mobiles[i].deployed_container] / self.mobiles[i].task[t][2] <= 1:
            #     offloading_strategy[i][0] = 1
            #     continue
            # 查询附近基站
            connect_server = self.mobiles[i].connected_server
            for container in self.server.deployed_container[connect_server]:
                if sigma[t][i][container + 1] == 1:
                    offloading_strategy[i][container + 1] = 1
                    is_offloaded = True
                    break
            if is_offloaded: continue
            # 查询其他基站
            for j in range(self.container_num + 1):
                if sigma[t][i][j] == 1:
                    offloading_strategy[i][j] = 1
                    is_offloaded = True
                    break
            if is_offloaded: continue                     
        #-------------取得FBAA的tau搜索上界----------------#
        tau_max = self.cal_tau(t, bandwidth_strategy, offloading_strategy, 0, False, sigma)
        return offloading_strategy, tau_max

    #########################APGTO+FGRA的对比算法函数###################
    # 考虑批处理等待延迟的带宽资源分配
    def FGRA(self, t, offloading_strategy) -> tuple([np.array, float]):
        bandwidth_strategy = None
        epsilon = 0.05
        # 求出每个任务的处理时间, 数据大小, 和有限传输时间
        k_m = np.zeros(self.container_num, dtype=int)
        data_size = np.zeros(self.mobile_num)
        wire_tran_time = np.ones(self.mobile_num)
        d_m = [10] * self.container_num                
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    k_m[j-1] += 1 
                    d_m[j-1] = np.min([self.mobiles[i].task[t][2], d_m[j-1]])  
        d_n = [self.mobiles[i].task[t][2] for i in range(self.mobile_num)]
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    d_n[i] = d_m[j-1]
        g_m =[self.server.inference_time[i][k_m[i]-1] for i in range(self.container_num)]
        process_time = np.ones([self.mobile_num])
        for i in range(self.mobile_num):
            for j in range(len(offloading_strategy[0])):  
                if offloading_strategy[i][j] == 1:
                    if j == 0:
                        process_time[i] = self.mobiles[i].local_inference_time[self.mobiles[i].deployed_container]
                    else:
                        process_time[i] = g_m[j-1]
                        data_size[i] = self.server.input_size[j-1]
                        wire_tran_time[i] = self.l_n_m[i][j-1]

        # 求出卸载到每个服务器的任务集合
        P_s = [[] for _ in range(self.server_num)]
        for i in range(self.mobile_num):
            for j in range(self.container_num):
                for k in range(self.server_num):
                    if offloading_strategy[i][j + 1] == 1 and j in self.server.deployed_container[k]:
                        P_s[k].append(i)
        
        # 算法核心
        bandwidth_strategy = np.zeros(self.mobile_num)
        G = np.ones(self.server_num)
        for i in range(self.server_num):
            g_min = 0
            for k in P_s[i]:
                g_min = np.max([g_min, process_time[i]/self.mobiles[i].task[t][2]])
            g_max = 1
            g = (g_min + g_max) / 2
            g_pre = 3
            total_B = 0
            while g_pre - g > epsilon:
                tmp_bandwidth_strategy = np.zeros(self.mobile_num)
                for j in P_s[i]:
                    if g * d_n[i]- process_time[j] - wire_tran_time[j] > 0:
                        r_n = data_size[j] / (g * d_n[i] - process_time[j] - wire_tran_time[j])
                        tmp_bandwidth_strategy[j] = self.cal_bandwidth_for_one(r_n, j, t)
                    else:
                        g_min = g
                        tmp_bandwidth_strategy = np.zeros(self.mobile_num)
                    total_B += bandwidth_strategy[j]
                    if total_B > self.server.total_bandwidth[i]:
                        g_min = g
                        tmp_bandwidth_strategy = np.zeros(self.mobile_num)
                if g_min != g:
                    g_max = g
                    for k in P_s[i]:
                        bandwidth_strategy[k] = tmp_bandwidth_strategy[k]
                g_pre = g
                g = (g_min + g_max) / 2
            G[i] = g
        G_max = np.max([G])
        for i in range(len(offloading_strategy)):
            if offloading_strategy[i][0] == 1:
                deployed_container = self.mobiles[i].deployed_container
                local_infer_time = self.mobiles[0].local_inference_time[deployed_container]
                task_deadline = self.mobiles[i].task[t][2]
                G_max = np.max([G_max, local_infer_time / task_deadline])
        return bandwidth_strategy, G_max
    
    # 随机将一个食物源的卸载策略修改为另一个的
    def EBP(self, t, FSP: List[FS]) -> List[FS]:
        for fs in FSP:
            fs_temp = FS(fs.bandwidth_strategy, fs.offloading_strategy, fs.G_m)
            fs_rand = fs
            while fs_rand == fs:
                fs_rand = random.choice(FSP)
            l = random.uniform(0, 0.5)
            n_l = self.mobile_num * l
            T_nl = random.sample([i for i in range(self.mobile_num)], int(n_l))
            for i in T_nl:
                for j in range(len(fs_temp.offloading_strategy[0])):
                    fs_temp.offloading_strategy[i][j] = fs_rand.offloading_strategy[i][j]
            fs_temp.bandwidth_strategy, fs_temp.G_m = self.FGRA(t, fs_temp.offloading_strategy)
            if fs_temp.G_m < fs.G_m:
                fs.bandwidth_strategy = fs_temp.bandwidth_strategy.copy()
                fs.offloading_strategy = [fs_temp.offloading_strategy[i].copy() for i in range(len(fs.offloading_strategy))]
                fs.G_m = fs_temp.G_m
        return FSP
    
    # 随机将一个食物源的卸载策略修改为表现最好的那一个
    def OBP(self, t, FSP: List[FS]) -> List[FS]:
        fs_best = FSP[0]
        for fs in FSP:
            if fs.G_m < fs_best.G_m:
                fs_best = fs
        for fs in FSP:
            fs_temp = FS(fs.bandwidth_strategy, fs.offloading_strategy, fs.G_m)
            l = random.uniform(0, 0.5)
            n_l = self.mobile_num * l
            T_nl = random.sample([i for i in range(self.mobile_num)], int(n_l))
            for i in T_nl:
                for j in range(len(fs_temp.offloading_strategy[0])):
                    fs_temp.offloading_strategy[i][j] = fs_best.offloading_strategy[i][j]
            fs_temp.bandwidth_strategy, fs_temp.G_m = self.FGRA(t, fs_temp.offloading_strategy)
            if fs_temp.G_m < fs.G_m:
                fs.bandwidth_strategy = fs_temp.bandwidth_strategy.copy()
                fs.offloading_strategy = [fs_temp.offloading_strategy[i].copy() for i in range(len(fs.offloading_strategy))]
                fs.G_m = fs_temp.G_m
        return FSP

    # 随机将一个食物源的卸载策略修改为卸载到其他可用服务器
    def SBP(self, t, FSP: List[FS], sigma, offload_to_local) -> List[FS]:
        for fs in FSP:
            fs_temp = FS(fs.bandwidth_strategy, fs.offloading_strategy, fs.G_m)
            l = random.uniform(0, 0.5)
            n_l = self.mobile_num * l
            T_nl = random.sample([i for i in range(self.mobile_num)], int(n_l))
            for i in T_nl:
                # 寻找可用卸载
                possible_offload = []
                for j in range(self.container_num + 1):
                    if(sigma[t][i][j] == 1):
                        if j != 0:
                            possible_offload.append(j)
                        else:
                            if offload_to_local:
                                possible_offload.append(0)
                select_offload = random.choice(possible_offload)
                # print(select_offload)
                # 清空原有卸载
                for k in range(self.container_num + 1):
                    fs_temp.offloading_strategy[i][k] = 0
                # 更新卸载策略             
                fs_temp.offloading_strategy[i][select_offload] = 1
            fs_temp.bandwidth_strategy, fs_temp.G_m = self.FGRA(t, fs_temp.offloading_strategy)
            if fs_temp.G_m < fs.G_m:
                fs.bandwidth_strategy = fs_temp.bandwidth_strategy.copy()
                fs.offloading_strategy = [fs_temp.offloading_strategy[i].copy() for i in range(len(fs.offloading_strategy))]
                fs.G_m = fs_temp.G_m
        return FSP 

    # 随机生成卸载策略
    def random_offload(self, t, sigma, offload_to_local) -> List[np.array]:
        random_offloading_strategy = np.zeros([self.mobile_num, self.container_num+1])

        for i in range(self.mobile_num):
            possible_offload = []
            for j in range(self.container_num + 1):
                if(sigma[t][i][j] == 1):
                    if j != 0:
                        possible_offload.append(j)
                    else:
                        if offload_to_local:
                            possible_offload.append(0)
            select_offload = random.choice(possible_offload)
            random_offloading_strategy[i][select_offload] = 1
        return random_offloading_strategy

    # 不考虑批处理等待延迟的启发式卸载策略和带宽分配策略
    def APGTO_FGRA(self, t, sigma, cost_in_this_slot, NP, cycle, offload_to_local) -> tuple([np.array, List[np.array], float]):
        FSP = []
        for _ in range(NP):
            temp_offloading_strategy = self.random_offload(t, sigma, offload_to_local)
            temp_bandwidth_strategy, G_m = self.FGRA(t, temp_offloading_strategy)
            fs  = FS(bandwidth_strategy=temp_bandwidth_strategy, offloading_strategy=temp_offloading_strategy, G_m=G_m)
            FSP.append(fs)
        # print(0)
        for _ in range(cycle):
            FSP = self.EBP(t=t, FSP=FSP)
            # print(1)
            FSP = self.OBP(t=t, FSP=FSP)
            # print(2)
            FSP = self.SBP(t=t, FSP=FSP, sigma=sigma, offload_to_local=offload_to_local)
            # print(3)

        G = 3
        for fs in FSP:
            if fs.G_m < G:
                G = fs.G_m
                fs_opt = fs
        #-------------取得精确tau----------------#
        tau_max = self.cal_tau(t, fs_opt.bandwidth_strategy, fs_opt.offloading_strategy, 0, False, sigma)
        return fs_opt.bandwidth_strategy, fs_opt.offloading_strategy, tau_max
    #########################APGTO+AVEB的对比算法函数################### 
    # 平均分配的带宽分配策略
    def AVEB_for_one(self, t, offloading_strategy, tau_max, cost_in_this_slot, sigma) -> tuple([np.array, float, float]):
        connect_mobile_num = np.zeros(self.server_num)
        bandwidth_strategy = np.zeros([self.mobile_num])
        for i in range(self.mobile_num):
            if offloading_strategy[i][0] == 0:
                connect_mobile_num[self.mobiles[i].connected_server] += 1
        for i in range(self.mobile_num):
            if offloading_strategy[i][0] == 0:
                bandwidth_strategy[i] = self.server.total_bandwidth[self.mobiles[i].connected_server] / connect_mobile_num[self.mobiles[i].connected_server]
        
        #-------------取得BFTUA的tau搜索上界tau_max以及精确tmp_tau----------------#
        tmp_tau = self.cal_tau_for_one(t, bandwidth_strategy, offloading_strategy, cost_in_this_slot, False, sigma)
        tau_max = self.cal_tau_for_one(t, bandwidth_strategy, offloading_strategy, 0, True, sigma)
        return bandwidth_strategy, tmp_tau, tau_max

    # 根据卸载策略和带宽分配策略求出tau
    def cal_tau_for_one(self, t, bandwidth_strategy, offloading_strategy, cost_in_this_slot, is_for_BFTUA, sigma) -> float:
        offloading_strategy = offloading_strategy
        bandwidth_strategy = bandwidth_strategy
        tau = 0
        k_m = np.zeros(self.container_num, dtype=int)
        g_m = np.zeros(self.container_num)
        d_n = [self.mobiles[i].task[t][2] for i in range(self.mobile_num)]
        for i in range(self.mobile_num):
            for j in range(1, len(offloading_strategy[0])):
                if offloading_strategy[i][j] == 1:
                    k_m[j-1] += 1
        g_m = [self.server.inference_time[i][k_m[i] - 1] for i in range(self.container_num)]
        # 获取成批时间
        batch_time = np.zeros(self.container_num)
        for i in range(self.mobile_num):
            wireless_tran_rate = self.environment.trans_rate(bandwidth_strategy[i], self.mobiles[i].trans_power, self.mobiles[i].distance, t)[self.isTrue]
            for j in range(self.container_num):
                if offloading_strategy[i][j+1]:
                    wireless_tran_time = self.environment.trans_time(wireless_tran_rate, self.server.input_size[j])
                    wire_tran_time = self.l_n_m[i][j]
                    batch_time[j] = np.max([batch_time[j], wireless_tran_time + wire_tran_time])
        # print("batch_time")
        # print(batch_time)
        
        for i in range(len(offloading_strategy)):
            if offloading_strategy[i][0] == 1:
                # 卸载到本地
                deployed_container = self.mobiles[i].deployed_container
                local_infer_time = self.mobiles[0].local_inference_time[deployed_container]
                task_deadline = self.mobiles[i].task[t][2]
                tau = np.max([tau, (local_infer_time + cost_in_this_slot) / task_deadline])
            else:
                # 卸载到服务器
                for j in range(self.container_num):
                    if offloading_strategy[i][j+1] == 1:
                        tau = np.max([tau, (batch_time[j] + g_m[j] + cost_in_this_slot) / d_n[i]])
            
        return tau
   
    # 随机将一个食物源的卸载策略修改为另一个的
    def EBP_AVEB(self, t, FSP: List[FS], sigma) -> List[FS]:
        for fs in FSP:
            fs_temp = FS(fs.bandwidth_strategy, fs.offloading_strategy, fs.G_m)
            fs_rand = fs
            while fs_rand == fs:
                fs_rand = random.choice(FSP)
            l = random.uniform(0, 0.5)
            n_l = self.mobile_num * l
            T_nl = random.sample([i for i in range(self.mobile_num)], int(n_l))
            for i in T_nl:
                for j in range(len(fs_temp.offloading_strategy[0])):
                    fs_temp.offloading_strategy[i][j] = fs_rand.offloading_strategy[i][j]
            fs_temp.bandwidth_strategy, fs_temp.G_m, tau_no_use = self.AVEB_for_one(t, fs_temp.offloading_strategy, 3, 0, sigma)
            if fs_temp.G_m < fs.G_m:
                fs.bandwidth_strategy = fs_temp.bandwidth_strategy.copy()
                fs.offloading_strategy = [fs_temp.offloading_strategy[i].copy() for i in range(len(fs.offloading_strategy))]
                fs.G_m = fs_temp.G_m
        return FSP
    
    # 随机将一个食物源的卸载策略修改为表现最好的那一个
    def OBP_AVEB(self, t, FSP: List[FS], sigma) -> List[FS]:
        fs_best = FSP[0]
        for fs in FSP:
            if fs.G_m < fs_best.G_m:
                fs_best = fs
        for fs in FSP:
            fs_temp = FS(fs.bandwidth_strategy, fs.offloading_strategy, fs.G_m)
            l = random.uniform(0, 0.5)
            n_l = self.mobile_num * l
            T_nl = random.sample([i for i in range(self.mobile_num)], int(n_l))
            for i in T_nl:
                for j in range(len(fs_temp.offloading_strategy[0])):
                    fs_temp.offloading_strategy[i][j] = fs_best.offloading_strategy[i][j]
            fs_temp.bandwidth_strategy, fs_temp.G_m, tau_no_use  = self.AVEB_for_one(t, fs_temp.offloading_strategy, 3, 0, sigma)
            if fs_temp.G_m < fs.G_m:
                fs.bandwidth_strategy = fs_temp.bandwidth_strategy.copy()
                fs.offloading_strategy = [fs_temp.offloading_strategy[i].copy() for i in range(len(fs.offloading_strategy))]
                fs.G_m = fs_temp.G_m
        return FSP

    # 随机将一个食物源的卸载策略修改为卸载到其他可用服务器
    def SBP_AVEB(self, t, FSP: List[FS], sigma, offload_to_local) -> List[FS]:
        for fs in FSP:
            fs_temp = FS(fs.bandwidth_strategy, fs.offloading_strategy, fs.G_m)
            l = random.uniform(0, 0.5)
            n_l = self.mobile_num * l
            T_nl = random.sample([i for i in range(self.mobile_num)], int(n_l))
            for i in T_nl:
                # 寻找可用卸载
                possible_offload = []
                for j in range(self.container_num + 1):
                    if(sigma[t][i][j] == 1):
                        if j != 0:
                            possible_offload.append(j)
                        else:
                            if offload_to_local:
                                possible_offload.append(0)
                select_offload = random.choice(possible_offload)
                # print(select_offload)
                # 清空原有卸载
                for k in range(self.container_num + 1):
                    fs_temp.offloading_strategy[i][k] = 0
                # 更新卸载策略             
                fs_temp.offloading_strategy[i][select_offload] = 1
            fs_temp.bandwidth_strategy, fs_temp.G_m, tau_no_use  = self.AVEB_for_one(t, fs_temp.offloading_strategy, 3, 0, sigma)
            if fs_temp.G_m < fs.G_m:
                fs.bandwidth_strategy = fs_temp.bandwidth_strategy.copy()
                fs.offloading_strategy = [fs_temp.offloading_strategy[i].copy() for i in range(len(fs.offloading_strategy))]
                fs.G_m = fs_temp.G_m
        return FSP 

    # 不考虑批处理等待延迟的启发式卸载策略
    def APGTO_AVEB(self, t, sigma, cost_in_this_slot, NP, cycle, offload_to_local) -> tuple([np.array, List[np.array], float]):
        FSP = []
        for _ in range(NP):
            temp_offloading_strategy = self.random_offload(t, sigma, offload_to_local)
            temp_bandwidth_strategy, G_m, tau_no_use = self.AVEB_for_one(t, temp_offloading_strategy, 3, 0, sigma)
            fs  = FS(bandwidth_strategy=temp_bandwidth_strategy, offloading_strategy=temp_offloading_strategy, G_m=G_m)
            FSP.append(fs)
        # print(0)
        for _ in range(cycle):
            FSP = self.EBP_AVEB(t=t, FSP=FSP, sigma=sigma)
            # print(1)
            FSP = self.OBP_AVEB(t=t, FSP=FSP, sigma=sigma)
            # print(2)
            FSP = self.SBP_AVEB(t=t, FSP=FSP, sigma=sigma, offload_to_local=offload_to_local)
            # print(3)

        G = 1000
        for fs in FSP:
            if fs.G_m < G:
                G = fs.G_m
                fs_opt = fs
        #-------------取得精确tau----------------#
        tau_max = self.cal_tau(t, fs_opt.bandwidth_strategy, fs_opt.offloading_strategy, 0, False, sigma)
        return fs_opt.bandwidth_strategy, fs_opt.offloading_strategy, tau_max
    
    #########################在线算法###################
    def online_algorithm(self) -> tuple([np.array, np.array]):
        # 决策变量
        offloading_strategy = []
        bandwidth_strategy = []
        # 算法开销
        alg_cost = []
        tau = []
        true_tau = []
        ave_tau = []
        cost = []
        serve_rate = []
        serve_rate_1 = []
        serve_rate_2 = []
        serve_rate_3 = []
        cost_in_this_slot = 0
        # 运行算法
        for T in tqdm(range(0, self.time_slot), desc="online: in different slot."):
            cost_in_this_slot = 0
            print(" ")
            sigma, d_m_min = self.gen_sigma()
            offloading_in_this_slot = []
            bandwidth_in_this_slot = []
            tau_in_this_slot = 3
            tmp_offloading_strategy=[]
            tmp_bandwidth_strategy = self.Average_allocate_bandwidth(self.server.total_bandwidth)
            tau_max = tau_in_this_slot
            start_time = time.time()
            for i in range(self.loop_k + self.loop_l):
                if(i < self.loop_k):
                    tmp_offloading_strategy, tau_max = self.BFTUA(T, tmp_bandwidth_strategy, sigma, self.alpha_appro, tau_max, False)
                else:
                    tmp_offloading_strategy, tau_max = self.BFTUA(T, tmp_bandwidth_strategy, sigma, self.alpha_appro, tau_max, True)
                if tau_max == 3: continue
                tmp_bandwidth_strategy, tmp_tau, tau_max = self.FBAA(T, tmp_offloading_strategy, tau_max, 0, sigma)
                if tmp_bandwidth_strategy is None: continue
                if(tmp_tau < tau_in_this_slot):                  
                    tau_in_this_slot = tmp_tau
                    offloading_in_this_slot = tmp_offloading_strategy
                    bandwidth_in_this_slot = tmp_bandwidth_strategy
            cost_in_this_slot = time.time()-start_time
            # 使满足卸载约束
            if(len(offloading_in_this_slot) == 0): offloading_in_this_slot = tmp_offloading_strategy
            if(len(bandwidth_in_this_slot) == 0): bandwidth_in_this_slot = tmp_bandwidth_strategy
            if(self.check_capacity(T, offloading_in_this_slot) == 0): offloading_in_this_slot = self.capacity_constaints(T, bandwidth_in_this_slot, offloading_in_this_slot, sigma) 
            offloading_in_this_slot = self.deadline_constaints(T, bandwidth_in_this_slot, offloading_in_this_slot, sigma, cost_in_this_slot)
            bandwidth_in_this_slot, _, _ = self.FBAA(T, offloading_in_this_slot, 3, 0, sigma)
            if(not self.cheak_offload(offloading_in_this_slot)): print("主算法卸载出错")
            # 求真实的tau
            tau_in_this_slot = self.cal_tau_for_true(T, bandwidth_in_this_slot, offloading_in_this_slot, 0, False, sigma)
            if (tau_in_this_slot > 1 and server_mode) :
                bandwidth_in_this_slot, offloading_in_this_slot = self.FBAAForOut(T, offloading_in_this_slot, 3, 0, sigma)
            tau_in_this_slot = self.cal_tau_for_true(T, bandwidth_in_this_slot, offloading_in_this_slot, 0, False, sigma)
            true_tau_in_this_slot = self.cal_tau_for_true(T, bandwidth_in_this_slot, offloading_in_this_slot, cost_in_this_slot, False, sigma)
            # ----------- record -----------------
            # 将该时隙的卸载策略和带宽分配和时间开销存进最终决策中
            serve_rate_in_this_slot, serve_rate_1_in_this_slot, serve_rate_2_in_this_slot, serve_rate_3_in_this_slot  = self.get_server_rate(T, offloading_in_this_slot, bandwidth_in_this_slot)
            offloading_strategy.append(offloading_in_this_slot.copy()) 
            bandwidth_strategy.append(bandwidth_in_this_slot.copy())  
            tau.append(tau_in_this_slot)
            alg_cost.append(cost_in_this_slot)
            serve_rate.append(serve_rate_in_this_slot)
            serve_rate_1.append(serve_rate_1_in_this_slot)
            serve_rate_2.append(serve_rate_2_in_this_slot)
            serve_rate_3.append(serve_rate_3_in_this_slot)
            # 返回结果
        ave_tau = np.average(tau)
        cost = np.average(alg_cost)
        rate = np.average(serve_rate)
        rate_1 = np.average(serve_rate_1)
        rate_2 = np.average(serve_rate_2)
        rate_3 = np.average(serve_rate_3)
        self.serve_rate_1 = rate_1
        self.serve_rate_2 = rate_2
        self.serve_rate_3 = rate_3
        self.serve_rate = rate                                            
        return ave_tau, cost

    # 平均分配带宽，公平卸载W
    def constract_algorithm_1(self) -> tuple([np.array, np.array]):
        # 决策变量
        offloading_strategy = []
        bandwidth_strategy = []
        # 算法开销
        serve_rate = []
        serve_rate_1 = []
        serve_rate_2 = []
        serve_rate_3 = []
        alg_cost = []
        tau = []
        cost_in_this_slot = 0
        # 运行算法
        for T in tqdm(range(0, self.time_slot), desc="online: in different slot."):
            cost_in_this_slot = 0
            sigma, d_m_min = self.gen_sigma()
            offloading_in_this_slot = []
            bandwidth_in_this_slot = []
            tau_in_this_slot = 3
            bandwidth_in_this_slot = self.Average_allocate_bandwidth(self.server.total_bandwidth)
            tau_max = tau_in_this_slot
            tmp_bandwidth_strategy = bandwidth_in_this_slot
            start_time = time.time()
            # print("-"*100)
            for i in range(self.loop_k + self.loop_l):
                # BFTUA_start_time = time.time()  
                if(i < self.loop_k):
                    # print(0)
                    tmp_offloading_strategy, tau_max = self.BFTUA(T, tmp_bandwidth_strategy, sigma, self.alpha_appro, tau_max, False)
                else:
                    # print(1)
                    tmp_offloading_strategy, tau_max = self.BFTUA(T, tmp_bandwidth_strategy, sigma, self.alpha_appro, tau_max, True)
                
                if tau_max == 3: continue
                tmp_bandwidth_strategy, tmp_tau, tau_max = self.AVEB(T, tmp_offloading_strategy,  tau_max, cost_in_this_slot, sigma)
                # print(2)
                if(tmp_tau < tau_in_this_slot):
                    # print(tmp_tau)
                    tau_in_this_slot = tmp_tau
                    offloading_in_this_slot = tmp_offloading_strategy
                    bandwidth_in_this_slot = tmp_bandwidth_strategy
            cost_in_this_slot = time.time()-start_time
            # 使满足卸载约束
            if(len(offloading_in_this_slot) == 0): offloading_in_this_slot = tmp_offloading_strategy
            if(len(bandwidth_in_this_slot) == 0): bandwidth_in_this_slot = tmp_bandwidth_strategy
            offloading_in_this_slot = self.capacity_constaints(T, bandwidth_in_this_slot, offloading_in_this_slot, sigma) 
            if(not self.cheak_offload(offloading_in_this_slot)):    print("not OK 1")
            # 求真实的tau
            tau_in_this_slot = self.cal_tau_for_true(T, bandwidth_in_this_slot, offloading_in_this_slot, 0, False, sigma)
            # ----------- record -----------------
            # 将该时隙的卸载策略和带宽分配和时间开销存进最终决策中
            serve_rate_in_this_slot, serve_rate_1_in_this_slot, serve_rate_2_in_this_slot, serve_rate_3_in_this_slot  = self.get_server_rate(T, offloading_in_this_slot, bandwidth_in_this_slot)
            offloading_strategy.append(offloading_in_this_slot.copy()) 
            bandwidth_strategy.append(bandwidth_in_this_slot.copy())  
            tau.append(tau_in_this_slot)
            alg_cost.append(cost_in_this_slot)
            serve_rate.append(serve_rate_in_this_slot)
            serve_rate_1.append(serve_rate_1_in_this_slot)
            serve_rate_2.append(serve_rate_2_in_this_slot)
            serve_rate_3.append(serve_rate_3_in_this_slot)
            # 返回结果
        ave_tau = np.average(tau)
        cost = np.average(alg_cost)
        rate = np.average(serve_rate)
        rate_1 = np.average(serve_rate_1)
        rate_2 = np.average(serve_rate_2)
        rate_3 = np.average(serve_rate_3)
        self.serve_rate_1 = rate_1
        self.serve_rate_2 = rate_2
        self.serve_rate_3 = rate_3
        self.serve_rate = rate                                            
        return ave_tau, cost
    
    # 公平分配带宽，就近卸载
    def constract_algorithm_2(self) -> tuple([np.array, np.array]):
        # 决策变量
        offloading_strategy = []
        bandwidth_strategy = []
        # 算法开销
        serve_rate = []
        serve_rate_1 = []
        serve_rate_2 = []
        serve_rate_3 = []
        alg_cost = []
        tau = []
        cost_in_this_slot = 0
        # 运行算法
        for T in tqdm(range(0, self.time_slot), desc="online: in different slot."):
            cost_in_this_slot = 0
            sigma, d_m_min = self.gen_sigma()
            offloading_in_this_slot = []
            bandwidth_in_this_slot = []
            tau_in_this_slot = 3
            tmp_offloading_strategy=[]
            tmp_bandwidth_strategy = self.Average_allocate_bandwidth(self.server.total_bandwidth)
            tau_max = tau_in_this_slot
            start_time = time.time()
            for i in range(self.loop_k + self.loop_l):
                if(i < self.loop_k):
                    tmp_offloading_strategy, tau_max = self.Nearest(T, tmp_bandwidth_strategy, sigma, tau_max)
                    # print(0," ", tau_max)
                else:
                    tmp_offloading_strategy, tau_max = self.Nearest(T, tmp_bandwidth_strategy, sigma, tau_max)
                    # print(1," ", tau_max)
                tmp_bandwidth_strategy, tmp_tau, tau_max = self.FBAA(T, tmp_offloading_strategy,  tau_max, cost_in_this_slot, sigma)
                if tmp_bandwidth_strategy is None: continue
                # print(2," ", tau_max)
                if(tmp_tau < tau_in_this_slot):                  
                    tau_in_this_slot = tmp_tau
                    offloading_in_this_slot = tmp_offloading_strategy
                    bandwidth_in_this_slot = tmp_bandwidth_strategy
            cost_in_this_slot = time.time()-start_time

            # 使满足卸载约束
            if(len(offloading_in_this_slot) == 0): offloading_in_this_slot = tmp_offloading_strategy
            if(len(bandwidth_in_this_slot) == 0): bandwidth_in_this_slot = tmp_bandwidth_strategy
            if(self.check_capacity(T, offloading_in_this_slot) == 0): offloading_in_this_slot = self.capacity_constaints(T, bandwidth_in_this_slot, offloading_in_this_slot, sigma) 
            # 求真实的tau
            if(not self.cheak_offload(offloading_in_this_slot)): print("算法2卸载出错")
            tau_in_this_slot = self.cal_tau_for_true(T, bandwidth_in_this_slot, offloading_in_this_slot, 0, False, sigma)
            # ----------- record -----------------
            # 将该时隙的卸载策略和带宽分配和时间开销存进最终决策中
            serve_rate_in_this_slot, serve_rate_1_in_this_slot, serve_rate_2_in_this_slot, serve_rate_3_in_this_slot  = self.get_server_rate(T, offloading_in_this_slot, bandwidth_in_this_slot)
            offloading_strategy.append(offloading_in_this_slot.copy()) 
            bandwidth_strategy.append(bandwidth_in_this_slot.copy())  
            tau.append(tau_in_this_slot)
            alg_cost.append(cost_in_this_slot)
            serve_rate.append(serve_rate_in_this_slot)
            serve_rate_1.append(serve_rate_1_in_this_slot)
            serve_rate_2.append(serve_rate_2_in_this_slot)
            serve_rate_3.append(serve_rate_3_in_this_slot)
            # 返回结果
        ave_tau = np.average(tau)
        cost = np.average(alg_cost)
        rate = np.average(serve_rate)
        rate_1 = np.average(serve_rate_1)
        rate_2 = np.average(serve_rate_2)
        rate_3 = np.average(serve_rate_3)
        self.serve_rate_1 = rate_1
        self.serve_rate_2 = rate_2
        self.serve_rate_3 = rate_3
        self.serve_rate = rate                                            
        return ave_tau, cost

    # APGTO+FGRA
    def constract_algorithm_3(self, NP:int, cycle:int) -> tuple([np.array, np.array]):
        # 决策变量
        offloading_strategy = []
        bandwidth_strategy = []
        # 算法开销
        serve_rate = []
        serve_rate_1 = []
        serve_rate_2 = []
        serve_rate_3 = []
        alg_cost = []
        tau = []
        cost_in_this_slot = 0
        # 运行算法
        for T in tqdm(range(0, self.time_slot), desc="online: in different slot."):
            cost_in_this_slot = 0
            print(' ')
            sigma, d_m_min = self.gen_sigma()
            start_time = time.time()
            bandwidth_in_this_slot, offloading_in_this_slot, tau_in_this_slot = self.APGTO_FGRA(T, sigma, cost_in_this_slot, NP, cycle, True)
            cost_in_this_slot = time.time()-start_time
            # 使满足卸载约束
            if(self.check_capacity(T, offloading_in_this_slot) == 0): offloading_in_this_slot = self.capacity_constaints(T, bandwidth_in_this_slot, offloading_in_this_slot, sigma) 
            # 求真实的tau
            tau_in_this_slot = self.cal_tau_for_true(T, bandwidth_in_this_slot, offloading_in_this_slot, 0, False, sigma)
            # ----------- record -----------------
            # 将该时隙的卸载策略和带宽分配和时间开销存进最终决策中
            serve_rate_in_this_slot, serve_rate_1_in_this_slot, serve_rate_2_in_this_slot, serve_rate_3_in_this_slot  = self.get_server_rate(T, offloading_in_this_slot, bandwidth_in_this_slot)
            offloading_strategy.append(offloading_in_this_slot.copy()) 
            bandwidth_strategy.append(bandwidth_in_this_slot.copy())  
            tau.append(tau_in_this_slot)
            alg_cost.append(cost_in_this_slot)
            serve_rate.append(serve_rate_in_this_slot)
            serve_rate_1.append(serve_rate_1_in_this_slot)
            serve_rate_2.append(serve_rate_2_in_this_slot)
            serve_rate_3.append(serve_rate_3_in_this_slot)
            # 返回结果
        ave_tau = np.average(tau)
        cost = np.average(alg_cost)
        rate = np.average(serve_rate)
        rate_1 = np.average(serve_rate_1)
        rate_2 = np.average(serve_rate_2)
        rate_3 = np.average(serve_rate_3)
        self.serve_rate_1 = rate_1
        self.serve_rate_2 = rate_2
        self.serve_rate_3 = rate_3
        self.serve_rate = rate                                            
        return ave_tau, cost

    # APGTO+平均
    def constract_algorithm_4(self, NP:int, cycle:int) -> tuple([np.array, np.array]):
        # 决策变量
        offloading_strategy = []
        bandwidth_strategy = []
        # 算法开销
        serve_rate = []
        serve_rate_1 = []
        serve_rate_2 = []
        serve_rate_3 = []
        alg_cost = []
        tau = []
        cost_in_this_slot = 0
        # 运行算法
        for T in tqdm(range(0, self.time_slot), desc="online: in different slot."):
            cost_in_this_slot = 0
            print(' ')
            sigma, d_m_min = self.gen_sigma()
            start_time = time.time()
            bandwidth_in_this_slot, offloading_in_this_slot, tau_in_this_slot = self.APGTO_AVEB(T, sigma, cost_in_this_slot, NP, cycle, True)
            cost_in_this_slot = time.time()-start_time
            # 使满足卸载约束
            if(self.check_capacity(T, offloading_in_this_slot) == 0): offloading_in_this_slot = self.capacity_constaints(T, bandwidth_in_this_slot, offloading_in_this_slot, sigma) 
            # 求真实的tau
            tau_in_this_slot = self.cal_tau_for_true(T, bandwidth_in_this_slot, offloading_in_this_slot, 0, False, sigma)
            # ----------- record -----------------
            # 将该时隙的卸载策略和带宽分配和时间开销存进最终决策中
            serve_rate_in_this_slot, serve_rate_1_in_this_slot, serve_rate_2_in_this_slot, serve_rate_3_in_this_slot  = self.get_server_rate(T, offloading_in_this_slot, bandwidth_in_this_slot)
            offloading_strategy.append(offloading_in_this_slot.copy()) 
            bandwidth_strategy.append(bandwidth_in_this_slot.copy())  
            tau.append(tau_in_this_slot)
            alg_cost.append(cost_in_this_slot)
            serve_rate.append(serve_rate_in_this_slot)
            serve_rate_1.append(serve_rate_1_in_this_slot)
            serve_rate_2.append(serve_rate_2_in_this_slot)
            serve_rate_3.append(serve_rate_3_in_this_slot)
            # 返回结果
        ave_tau = np.average(tau)
        cost = np.average(alg_cost)
        rate = np.average(serve_rate)
        rate_1 = np.average(serve_rate_1)
        rate_2 = np.average(serve_rate_2)
        rate_3 = np.average(serve_rate_3)
        self.serve_rate_1 = rate_1
        self.serve_rate_2 = rate_2
        self.serve_rate_3 = rate_3
        self.serve_rate = rate                                            
        return ave_tau, cost

    # 贪婪分配带宽， 公平卸载
    def constract_algorithm_5(self) -> tuple([np.array, np.array]):
        # 决策变量
        offloading_strategy = []
        bandwidth_strategy = []
        # 算法开销
        serve_rate = []
        serve_rate_1 = []
        serve_rate_2 = []
        serve_rate_3 = []
        alg_cost = []
        tau = []
        cost_in_this_slot = 0
        # 运行算法
        for T in tqdm(range(0, self.time_slot), desc="online: in different slot."):
            cost_in_this_slot = 0
            tau_max = 3
            sigma, d_m_min = self.gen_sigma()
            offloading_in_this_slot = []
            bandwidth_in_this_slot = []
            bandwidth_in_this_slot = self.Average_allocate_bandwidth(self.server.total_bandwidth)
            tmp_bandwidth_strategy = bandwidth_in_this_slot
            start_time = time.time()
            # print("-"*100)
            for i in range(1):
                # BFTUA_start_time = time.time()        
                offloading_in_this_slot, tau_max = self.BFTUA(T, tmp_bandwidth_strategy, sigma, self.alpha_appro, tau_max, True)
                bandwidth_in_this_slot, tmp_tau, tau_max = self.greedyB(T, offloading_in_this_slot,  tau_max, cost_in_this_slot, sigma)
                # print(2)
                tau_in_this_slot = tmp_tau
            cost_in_this_slot = time.time()-start_time
            # 使满足卸载约束
            
            if(self.check_capacity(T, offloading_in_this_slot) == 0): offloading_in_this_slot = self.capacity_constaints(T, bandwidth_in_this_slot, offloading_in_this_slot, sigma) 
            # 求真实的tau
            tau_in_this_slot = self.cal_tau_for_true(T, bandwidth_in_this_slot, offloading_in_this_slot, 0, False, sigma)
            server_rate_in_this_slot = self.get_server_rate(T, offloading_in_this_slot, bandwidth_in_this_slot)
            # ----------- record -----------------
            # 将该时隙的卸载策略和带宽分配和时间开销存进最终决策中
            serve_rate_in_this_slot, serve_rate_1_in_this_slot, serve_rate_2_in_this_slot, serve_rate_3_in_this_slot  = self.get_server_rate(T, offloading_in_this_slot, bandwidth_in_this_slot)
            offloading_strategy.append(offloading_in_this_slot.copy()) 
            bandwidth_strategy.append(bandwidth_in_this_slot.copy())  
            tau.append(tau_in_this_slot)
            alg_cost.append(cost_in_this_slot)
            serve_rate.append(serve_rate_in_this_slot)
            serve_rate_1.append(serve_rate_1_in_this_slot)
            serve_rate_2.append(serve_rate_2_in_this_slot)
            serve_rate_3.append(serve_rate_3_in_this_slot)
            # 返回结果
        ave_tau = np.average(tau)
        cost = np.average(alg_cost)
        rate = np.average(serve_rate)
        rate_1 = np.average(serve_rate_1)
        rate_2 = np.average(serve_rate_2)
        rate_3 = np.average(serve_rate_3)
        self.serve_rate_1 = rate_1
        self.serve_rate_2 = rate_2
        self.serve_rate_3 = rate_3
        self.serve_rate = rate                                            
        return ave_tau, cost
    
    # 贪婪卸载， 公平带宽
    def constract_algorithm_6(self) -> tuple([np.array, np.array]):
        # 决策变量
        offloading_strategy = []
        bandwidth_strategy = []
        # 算法开销
        serve_rate = []
        serve_rate_1 = []
        serve_rate_2 = []
        serve_rate_3 = []
        alg_cost = []
        tau = []
        ave_tau = []
        cost = []
        cost_in_this_slot = 0
        # 运行算法
        for T in tqdm(range(0, self.time_slot), desc="online: in different slot."):
            cost_in_this_slot = 0
            print(" ")
            sigma, d_m_min = self.gen_sigma()
            offloading_in_this_slot = []
            bandwidth_in_this_slot = []
            tau_in_this_slot = 3
            tmp_offloading_strategy=np.zeros([self.mobile_num, self.container_num + 1])
            tmp_bandwidth_strategy = self.Average_allocate_bandwidth(self.server.total_bandwidth)
            tau_max = tau_in_this_slot
            start_time = time.time()
            for i in range(self.loop_k + self.loop_l):
                if(i < self.loop_k):
                    tmp_offloading_strategy, tau_max = self.greedyO(T, tmp_bandwidth_strategy, sigma, self.alpha_appro, tau_max, False)
                else:
                    tmp_offloading_strategy, tau_max = self.greedyO(T, tmp_bandwidth_strategy, sigma, self.alpha_appro, tau_max, True)
                if tmp_offloading_strategy is None: continue
                tmp_bandwidth_strategy, tmp_tau, tau_max = self.FBAA(T, tmp_offloading_strategy,  tau_max, cost_in_this_slot, sigma)
                if tmp_bandwidth_strategy is None: continue
                if(tmp_tau < tau_in_this_slot):                  
                    tau_in_this_slot = tmp_tau
                    # print(tau_in_this_slot)
                    offloading_in_this_slot = tmp_offloading_strategy
                    bandwidth_in_this_slot = tmp_bandwidth_strategy
            cost_in_this_slot = time.time()-start_time
            # 使满足卸载约束
            if (len(offloading_in_this_slot) == 0): offloading_in_this_slot = tmp_offloading_strategy
            if (len(bandwidth_in_this_slot) == 0): bandwidth_in_this_slot = tmp_bandwidth_strategy
            if(self.check_capacity(T, offloading_in_this_slot) == 0): offloading_in_this_slot = self.capacity_constaints(T, bandwidth_in_this_slot, offloading_in_this_slot, sigma) 
            # 求真实的tau
            tau_in_this_slot = self.cal_tau_for_true(T, bandwidth_in_this_slot, offloading_in_this_slot, 0, False, sigma)
            # ----------- record -----------------
            # 将该时隙的卸载策略和带宽分配和时间开销存进最终决策中
            serve_rate_in_this_slot, serve_rate_1_in_this_slot, serve_rate_2_in_this_slot, serve_rate_3_in_this_slot  = self.get_server_rate(T, offloading_in_this_slot, bandwidth_in_this_slot)
            offloading_strategy.append(offloading_in_this_slot.copy()) 
            bandwidth_strategy.append(bandwidth_in_this_slot.copy())  
            tau.append(tau_in_this_slot)
            alg_cost.append(cost_in_this_slot)
            serve_rate.append(serve_rate_in_this_slot)
            serve_rate_1.append(serve_rate_1_in_this_slot)
            serve_rate_2.append(serve_rate_2_in_this_slot)
            serve_rate_3.append(serve_rate_3_in_this_slot)
            # 返回结果
        ave_tau = np.average(tau)
        cost = np.average(alg_cost)
        rate = np.average(serve_rate)
        rate_1 = np.average(serve_rate_1)
        rate_2 = np.average(serve_rate_2)
        rate_3 = np.average(serve_rate_3)
        self.serve_rate_1 = rate_1
        self.serve_rate_2 = rate_2
        self.serve_rate_3 = rate_3
        self.serve_rate = rate                                            
        return ave_tau, cost

# 最大流的辅助类   
class Flow:
    def __init__(self, to = -1, next = -1, cap = -1):
        self.to = to
        self.next = next
        self.cap = cap

# 最大流算法
class Maxflow:
    def __init__(self, res: List[np.array], mobile_num: int, container_num: int):
        self.origin_res = res.copy()
        self.point_num = len(res)
        self.mobile_num = mobile_num
        # print(mobile_num)
        self.container_num = container_num
        self.deep = [-1] * self.point_num
        self.head = [-1] * self.point_num
        self.cur = [-1] * self.point_num
        self.cnt = -1
        self.q = queue.Queue()
        self.edge = [Flow() for _ in range(self.point_num*self.point_num)]
    
    def addEdge(self, u, v, cost):
        self.cnt += 1
        self.edge[self.cnt].to = v
        self.edge[self.cnt].cap = cost
        self.edge[self.cnt].next = self.head[u]
        self.head[u] = self.cnt
        # 反向建边
        self.cnt += 1
        self.edge[self.cnt].to = u
        self.edge[self.cnt].cap = 0
        self.edge[self.cnt].next = self.head[v]
        self.head[v] = self.cnt

    def bfs(self) -> bool:
        while(not self.q.empty()):  self.q.get()
        self.cur = self.head.copy()
        self.deep = [-1] * self.point_num
        self.deep[0] = 0
        self.q.put(0)
        while(not self.q.empty()):
            x = self.q.get()
            i = self.head[x]

            while(i != -1):
                u = self.edge[i].to
                cap = self.edge[i].cap
                if(cap > 0 and self.deep[u] == -1):
                    self.deep[u] = self.deep[x] + 1
                    self.q.put(u)
                i = self.edge[i].next 
        if(self.deep[-1] > 0): 
            return True
        else:   return False
    
    def dfs(self, x: int, mx: int) -> int:
        if(x == self.point_num - 1):
            return mx
        i = self.cur[x]
        while(i != -1):
            self.cur[x] = i
            u = self.edge[i].to
            cap = self.edge[i].cap
            if cap > 0 and self.deep[u] == self.deep[x] + 1:
                a = self.dfs(u, np.min([cap, mx]))
                if a != 0:
                    self.edge[i].cap -= a
                    self.edge[i^1].cap += a
                    return a
            i = self.edge[i].next
        return 0

               
    def dicnic(self) -> tuple([int, np.array]):
        for i in range(len(self.origin_res)-1, -1, -1):
            for j in range(len(self.origin_res[0])-1, -1, -1):
                if self.origin_res[i][j] > 0:
                    self.addEdge(i, j, self.origin_res[i][j])
        maxflow = 0
        x_n_m = np.zeros([self.mobile_num, self.container_num], dtype=int)
        while(self.bfs()):
            while(1):
                f = self.dfs(0, np.inf)
                if(f == 0): break
                maxflow += f   
        for i in range(1, len(self.edge),2):
            if self.edge[i].cap > 0:
                u = self.edge[i].to
                v = self.edge[i-1].to
                if u in range(1, self.mobile_num + 1) and v in range(self.mobile_num + 1, self.mobile_num + self.container_num + 1):
                    x_n_m[u - 1][v - self.mobile_num - 1] = 1
        return maxflow, x_n_m




####################################此处是结束的分界线^_^###########################################