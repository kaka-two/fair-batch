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

 
    #########################简单的对比算法函数###################
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

   #########################APGTO+AVEB的对比算法函数################### 
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
    

    # Leading+Average
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
            # cost_in_this_slot = 0
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
    
    # Nearest+FAST
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
            # cost_in_this_slot = 0
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

    # APGTO+Average
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
            # cost_in_this_slot = 0
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

    # Leading+Average
    def constract_algorithm_4(self) -> tuple([np.array, np.array]):
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
            # cost_in_this_slot = 0
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
    
    # G-offlead+FAST
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
        ave_tau = []
        cost = []
        cost_in_this_slot = 0
        # 运行算法
        for T in tqdm(range(0, self.time_slot), desc="online: in different slot."):
            # cost_in_this_slot = 0
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


class Flow:
    def __init__(self, to = -1, next = -1, cap = -1):
        self.to = to
        self.next = next
        self.cap = cap

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

