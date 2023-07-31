import random
import numpy as np
import env
import algo
from tqdm import tqdm
import matplotlib.pyplot as plt
random.seed(10000)

time_slot = 10
server_num = 2
container_num = 18
alpha_appro = 2
container_type = 3
serve_rate = []
# 指定条件的实验
def fix(mobile_num = 60, deadline = 0.4,  loop_k = 1, loop_l = 0):
    server = env.Server(server_num=server_num)
    environmet = env.Environment()
    mobiles = [env.Mobile(server_num=server_num, mobile_num=mobile_num,  deadline=deadline) for _ in range(mobile_num)]
    for _ in range(1):
        # 开始实验
        ALG = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 0)
        ALG_true = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 1)
        tau_0, cost_0 = ALG.online_algorithm()
        tau_true_0, cost_true_0 = ALG_true.online_algorithm()
        serve_rate.append(ALG.server_rate)        
        # tau_1, cost_1 = ALG.constract_algorithm_1()
        # serve_rate.append(ALG.server_rate) 
        # tau_2, cost_2 = ALG.constract_algorithm_2()
        # serve_rate.append(ALG.server_rate)
        # tau_3, cost_3 = ALG.constract_algorithm_4(NP=20, cycle=10)
        # serve_rate.append(ALG.server_rate) 
        # tau_4, cost_4 = ALG.constract_algorithm_6()
        # serve_rate.append(ALG.server_rate) 
        print("服务器数量: ", server_num, ", 用户数量: ", mobile_num)
        print("公平卸载，公平带宽: ", round(tau_0, 2), ", 算法开销: ", round(cost_0 * 1000, 2))
        print("公平卸载，公平带宽: ", round(tau_true_0, 2), ", 算法开销: ", round(cost_true_0 * 1000, 2))
        # print("公平卸载，平均带宽: ", round(tau_1, 2), ", 算法开销: ",round(cost_1 * 1000, 2))
        # print("就近卸载，公平带宽: ", round(tau_2, 2), ", 算法开销: ",round(cost_2 * 1000, 2))
        # # print("离线对比算法: ", round(tau_3, 2))
        # print("贪婪卸载，公平带宽: ", round(tau_4, 2), ", 算法开销: ",round(cost_4 * 1000, 2))
        # for i in range(len(serve_rate)):
        #     print(i, np.round(serve_rate[i], 2))

# 随着loop增加的对比实验
def exp_with_loop(server_num = 4, mobile_num = 60, deadline = 0.4, loop_k = 1, loop_l = 0):
    repeat = 10
    tau = np.zeros([1, 5])
    cost = np.zeros([1, 5])
    tau_temp = np.zeros(7)
    cost_temp = np.zeros(7)

    for _ in range(repeat):
        j = -1
        environmet = env.Environment()
        server = env.Server(server_num=server_num)
        mobiles = [env.Mobile(server_num=server_num, mobile_num=mobile_num, deadline=deadline) for _ in range(mobile_num)]
        for loop_k in range(1, 11, 2):
            j += 1
            # 开始实验
            ALG = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 0)
            tau_temp[0], cost_temp[0] = ALG.online_algorithm()
            for i in range(1):
                tau[i][j] += tau_temp[i]
                cost[i][j] += cost_temp[i]
    for i in range(1):
        for j in range(5):
            tau[i][j] = round(tau[i][j] / repeat, 2)
            cost[i][j] = round(cost[i][j] / repeat, 2)
    return tau, cost

# 随着用户设备增加的对比实验
def exp_with_mobile_num(server_num = 4, mobile_num = 60, deadline = 0.4, loop_k = 1, loop_l = 0):
    repeat = 1
    tau = np.zeros([7, 5])
    cost = np.zeros([7, 5])
    tau_temp = np.zeros(7)
    cost_temp = np.zeros(7)

    for _ in tqdm(range(0, repeat), desc="exp_with_mobile_num"):
        j = -1
        environmet = env.Environment()
        server = env.Server(server_num=server_num)
        for mobile_num in range(20, 120, 20):
            j += 1
            mobiles = [env.Mobile(server_num=server_num, mobile_num=mobile_num, deadline=deadline) for _ in range(mobile_num)]
            # 开始实验
            ALG = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 0)
            ALG_true = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 1)
            tau_temp[0], cost_temp[0] = ALG.constract_algorithm_2()
            tau_temp[1], cost_temp[1] = ALG.constract_algorithm_1()
            # tau_temp[2], cost_temp[2] = ALG.constract_algorithm_4(NP=20, cycle=50)
            tau_temp[3], cost_temp[3] = ALG.constract_algorithm_6()
            tau_temp[4], cost_temp[4] = ALG_true.constract_algorithm_6()
            tau_temp[5], cost_temp[5] = ALG.online_algorithm()
            tau_temp[6], cost_temp[6] = ALG_true.online_algorithm()
            for i in range(7):
                tau[i][j] += tau_temp[i]
                cost[i][j] += cost_temp[i]
    for i in range(7):
        for j in range(5):
            tau[i][j] = round(tau[i][j] / repeat, 2)
            cost[i][j] *= 1000
            cost[i][j] = round(cost[i][j] / repeat, 2)
    return tau, cost

# 随着用户设备增加的对比实验
def exp_with_bandwidth(server_num = 4, mobile_num = 60, deadline = 0.4, loop_k = 1, loop_l = 0):
    repeat = 5
    tau = np.zeros([7, 5])
    cost = np.zeros([7, 5])
    tau_temp = np.zeros(7)
    cost_temp = np.zeros(7)

    for _ in tqdm(range(0, repeat), desc="exp_with_bandwidth"):
        j = -1
        environmet = env.Environment()
        server = env.Server(server_num=server_num)
        mobiles = [env.Mobile(server_num=server_num, mobile_num=mobile_num, deadline=deadline) for _ in range(mobile_num)]
        for band in range(16, 36, 4):
            j += 1
            server.re_init(server_num = server_num, band = band)
            # 开始实验
            ALG = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 0)
            ALG_true = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 1)
            tau_temp[0], cost_temp[0] = ALG.constract_algorithm_2()
            tau_temp[1], cost_temp[1] = ALG.constract_algorithm_1()
            tau_temp[2], cost_temp[2] = ALG.constract_algorithm_4(NP=20, cycle=50)
            tau_temp[3], cost_temp[3] = ALG.constract_algorithm_6()
            tau_temp[4], cost_temp[4] = ALG_true.constract_algorithm_6()
            tau_temp[5], cost_temp[5] = ALG.online_algorithm()
            tau_temp[6], cost_temp[6] = ALG_true.online_algorithm()
            for i in range(7):
                tau[i][j] += tau_temp[i]
                cost[i][j] += cost_temp[i]
    for i in range(7):
        for j in range(5):
            tau[i][j] = round(tau[i][j] / repeat, 2)
            cost[i][j] *= 1000
            cost[i][j] = round(cost[i][j] / repeat, 2)
    return tau, cost

# 随着deadline波动范围增加的对比实验
def exp_with_deadline(server_num = 4, mobile_num = 60, deadline = 0.4, loop_k = 1, loop_l = 0):
    repeat = 5
    tau = np.zeros([7, 5])
    cost = np.zeros([7, 5])
    tau_temp = np.zeros(7)
    cost_temp = np.zeros(7)
    
    for _ in tqdm(range(0, repeat), desc="exp_with_deadline"):
        j = -1
        environmet = env.Environment()
        server = env.Server(server_num=server_num)
        mobiles = [env.Mobile(server_num=server_num, mobile_num=mobile_num, deadline=deadline) for _ in range(mobile_num)]
        for deadline in range(300, 550, 50):
            j += 1
            deadline /= 1000
            for mobile in mobiles:
                mobile.re_init_deadline(deadline)
            # 开始实验
            ALG = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 0)
            ALG_true = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 1)
            tau_temp[0], cost_temp[0] = ALG.constract_algorithm_2()
            tau_temp[1], cost_temp[1] = ALG.constract_algorithm_1()
            tau_temp[2], cost_temp[2] = ALG.constract_algorithm_4(NP=20, cycle=50)
            tau_temp[3], cost_temp[3] = ALG.constract_algorithm_6()
            tau_temp[4], cost_temp[4] = ALG_true.constract_algorithm_6()
            tau_temp[5], cost_temp[5] = ALG.online_algorithm()
            tau_temp[6], cost_temp[6] = ALG_true.online_algorithm()
            for i in range(7):
                tau[i][j] += tau_temp[i]
                cost[i][j] += cost_temp[i]
    for i in range(7):
        for j in range(5):
            tau[i][j] = round(tau[i][j] / repeat, 2)
            cost[i][j] *= 1000
            cost[i][j] = round(cost[i][j] / repeat, 2)
    return tau, cost

# 随着服务器增加的对比实验
def exp_with_server_num(server_num = 4, mobile_num = 60, deadline = 0.4, loop_k = 1, loop_l = 0):
    repeat = 1
    tau = np.zeros([7, 5])
    cost = np.zeros([7, 5])
    tau_temp = np.zeros(7)
    cost_temp = np.zeros(7)

    for _ in tqdm(range(0, repeat), desc="exp_with_server_num"):
        j = -1
        environmet = env.Environment()
        server = env.Server(server_num=server_num)
        mobiles = [env.Mobile(server_num=server_num, mobile_num=mobile_num, deadline=deadline) for _ in range(mobile_num)]
        for server_num in range(2, 7, 1):
            j += 1
            server.re_init(server_num=server_num)
            for mobile in mobiles:
                mobile.re_init(server_num=server_num)
            # 开始实验
            ALG = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 0)
            ALG_true = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 1)
            tau_temp[0], cost_temp[0] = ALG.constract_algorithm_2()
            tau_temp[1], cost_temp[1] = ALG.constract_algorithm_1()
            # tau_temp[2], cost_temp[2] = ALG.constract_algorithm_4(NP=20, cycle=50)
            tau_temp[3], cost_temp[3] = ALG.constract_algorithm_6()
            tau_temp[4], cost_temp[4] = ALG_true.constract_algorithm_6()
            tau_temp[5], cost_temp[5] = ALG.online_algorithm()
            tau_temp[6], cost_temp[6] = ALG_true.online_algorithm()
            for i in range(7):
                tau[i][j] += tau_temp[i]
                cost[i][j] += cost_temp[i]
    for i in range(7):
        for j in range(5):
            tau[i][j] = round(tau[i][j] / repeat, 2)
            cost[i][j] *= 1000
            cost[i][j] = round(cost[i][j] / repeat, 2)
    return tau, cost

# 服务率-终端设备对比实验
def serve_rate_with_mobile_num(server_num = 4, mobile_num = 150, deadline = 0.4, loop_k = 1, loop_l = 0):
    repeat = 1
    serve_rate = np.zeros([7, 5])
    serve_rate_1 = np.zeros([7, 5])
    serve_rate_2 = np.zeros([7, 5])
    serve_rate_3 = np.zeros([7, 5])
    for _ in tqdm(range(0, repeat), desc="serve_rate_with_mobile_num"):
        j = -1
        server = env.Server(server_num=server_num)
        environmet = env.Environment()
        for mobile_num in range(120, 220, 20):
            j += 1
            mobiles = [env.Mobile(server_num=server_num, mobile_num=mobile_num, deadline=deadline) for _ in range(mobile_num)]
            # 开始实验
            ALG = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 0)
            ALG_true = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 1)
            ALG.constract_algorithm_2()
            serve_rate[0][j] += ALG.serve_rate
            serve_rate_1[0][j] += ALG.serve_rate_1
            serve_rate_2[0][j] += ALG.serve_rate_2
            serve_rate_3[0][j] += ALG.serve_rate_3
            ALG.constract_algorithm_1()
            serve_rate[1][j] += ALG.serve_rate
            serve_rate_1[1][j] += ALG.serve_rate_1
            serve_rate_2[1][j] += ALG.serve_rate_2
            serve_rate_3[1][j] += ALG.serve_rate_3
            ALG.constract_algorithm_4(NP=20, cycle=50)
            serve_rate[2][j] += ALG.serve_rate
            serve_rate_1[2][j] += ALG.serve_rate_1
            serve_rate_2[2][j] += ALG.serve_rate_2
            serve_rate_3[2][j] += ALG.serve_rate_3
            ALG.constract_algorithm_6()
            serve_rate[3][j] += ALG.serve_rate
            serve_rate_1[3][j] += ALG.serve_rate_1
            serve_rate_2[3][j] += ALG.serve_rate_2
            serve_rate_3[3][j] += ALG.serve_rate_3
            ALG_true.constract_algorithm_6()
            serve_rate[4][j] += ALG.serve_rate
            serve_rate_1[4][j] += ALG.serve_rate_1
            serve_rate_2[4][j] += ALG.serve_rate_2
            serve_rate_3[4][j] += ALG.serve_rate_3
            ALG.online_algorithm()
            serve_rate[5][j] += ALG.serve_rate
            serve_rate_1[5][j] += ALG.serve_rate_1
            serve_rate_2[5][j] += ALG.serve_rate_2
            serve_rate_3[5][j] += ALG.serve_rate_3
            ALG_true.online_algorithm()
            serve_rate[6][j] += ALG.serve_rate
            serve_rate_1[6][j] += ALG.serve_rate_1
            serve_rate_2[6][j] += ALG.serve_rate_2
            serve_rate_3[6][j] += ALG.serve_rate_3
    for i in range(7):
        for j in range(5):
            serve_rate[i][j] = round(serve_rate[i][j] / repeat, 2)
            serve_rate_1[i][j] = round(serve_rate_1[i][j] / repeat, 2)
            serve_rate_2[i][j] = round(serve_rate_2[i][j] / repeat, 2)
            serve_rate_3[i][j] = round(serve_rate_3[i][j] / repeat, 2)
    return serve_rate, serve_rate_1, serve_rate_2, serve_rate_3

# 服务率-服务器对比实验
def serve_rate_with_server_num(server_num = 4, mobile_num = 150, deadline = 0.4, loop_k = 1, loop_l = 0):
    repeat = 1
    serve_rate = np.zeros([7, 5])
    serve_rate_1 = np.zeros([7, 5])
    serve_rate_2 = np.zeros([7, 5])
    serve_rate_3 = np.zeros([7, 5])
    for _ in tqdm(range(0, repeat), desc="serve_rate_with_server_num"):
        j = -1
        server = env.Server(server_num=server_num)
        environmet = env.Environment()
        mobiles = [env.Mobile(server_num=server_num, mobile_num=mobile_num, deadline=deadline) for _ in range(mobile_num)]
        for server_num in range(2, 7, 1):
            j += 1
            server.re_init(server_num=server_num)
            for mobile in mobiles:
                mobile.re_init(server_num=server_num)
            # 开始实验
            ALG = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 0)
            ALG_true = algo.My_algo(server = server, environment = environmet, mobiles = mobiles, loop_k = loop_k, loop_l = loop_l, server_num = server_num, container_num = server_num * 3, time_slot = time_slot, alpha_appro = alpha_appro, isTrue = 1)
            ALG.constract_algorithm_2()
            serve_rate[0][j] += ALG.serve_rate
            serve_rate_1[0][j] += ALG.serve_rate_1
            serve_rate_2[0][j] += ALG.serve_rate_2
            serve_rate_3[0][j] += ALG.serve_rate_3
            ALG.constract_algorithm_1()
            serve_rate[1][j] += ALG.serve_rate
            serve_rate_1[1][j] += ALG.serve_rate_1
            serve_rate_2[1][j] += ALG.serve_rate_2
            serve_rate_3[1][j] += ALG.serve_rate_3
            ALG.constract_algorithm_4(NP=20, cycle=50)
            serve_rate[2][j] += ALG.serve_rate
            serve_rate_1[2][j] += ALG.serve_rate_1
            serve_rate_2[2][j] += ALG.serve_rate_2
            serve_rate_3[2][j] += ALG.serve_rate_3
            ALG.constract_algorithm_6()
            serve_rate[3][j] += ALG.serve_rate
            serve_rate_1[3][j] += ALG.serve_rate_1
            serve_rate_2[3][j] += ALG.serve_rate_2
            serve_rate_3[3][j] += ALG.serve_rate_3
            ALG_true.constract_algorithm_6()
            serve_rate[4][j] += ALG.serve_rate
            serve_rate_1[4][j] += ALG.serve_rate_1
            serve_rate_2[4][j] += ALG.serve_rate_2
            serve_rate_3[4][j] += ALG.serve_rate_3
            ALG.online_algorithm()
            serve_rate[5][j] += ALG.serve_rate
            serve_rate_1[5][j] += ALG.serve_rate_1
            serve_rate_2[5][j] += ALG.serve_rate_2
            serve_rate_3[5][j] += ALG.serve_rate_3
            ALG_true.online_algorithm()
            serve_rate[6][j] += ALG.serve_rate_1
            serve_rate_1[6][j] += ALG.serve_rate_1
            serve_rate_2[6][j] += ALG.serve_rate_2
            serve_rate_3[6][j] += ALG.serve_rate_3
    for i in range(7):
        for j in range(5):
            serve_rate[i][j] = round(serve_rate[i][j] / repeat, 2)
            serve_rate_1[i][j] = round(serve_rate_1[i][j] / repeat, 2)
            serve_rate_2[i][j] = round(serve_rate_2[i][j] / repeat, 2)
            serve_rate_3[i][j] = round(serve_rate_3[i][j] / repeat, 2)
    return serve_rate, serve_rate_1, serve_rate_2, serve_rate_3

# 记录tau
def record(result, c_result, exp_name):
    with open("record.txt", "a+") as f:
        f.write("\n")
        f.write(exp_name)
        f.write("\n")
        f.write("tau:\n")
    for i in range(len(result)):
        with open("record.txt", "a+") as f:
            f.write("[")
            for j in range(len(result[0])):
                f.write(str(result[i][j]))
                f.write(", ")
            f.write("]")
            f.write(", ")
            f.write("\n")
    with open("record.txt", "a+") as f:
        f.write("\n")
        f.write("cost:\n")
    for i in range(len(c_result)):
        with open("record.txt", "a+") as f:
            f.write("[")
            for j in range(len(c_result[0])):
                f.write(str(c_result[i][j]))
                f.write(", ")
            f.write("]")
            f.write(", ")
            f.write("\n")
    # # 绘图
    # plt.ylim([0, 2])
    # plt.ylabel("tau")
    # for i in range(len(result)):
    #     plt.plot(result[i], label=str(i))
    # plt.legend()
    # plt.show()
    # plt.ylim([0, 0.2])
    # for i in range(len(c_result)):
    #     plt.plot(c_result[i], label=str(i))
    # plt.legend()
    # plt.show()

# 记录服务率
def record_serve_rate(serve_rate, exp_name):
    with open("record.txt", "a+") as f:
        f.write("\n")
        f.write(exp_name)
        f.write("\n")
        f.write("serve_rate:\n")
    for i in range(len(serve_rate)):
        with open("record.txt", "a+") as f:
            f.write(str(serve_rate[i]))
            f.write("\n")
    # # 绘图
    # plt.ylabel("serve_rate")
    # for i in range(len(serve_rate)):
    #     plt.plot(serve_rate[i], label=str(i))
    # plt.legend()
    # plt.show()

# 运行入口
if __name__ == "__main__":
    # fix()

    # # loop实验
    # result, c_result = exp_with_loop()
    # record(result, c_result, "exp_with_loop")

    # 终端设备数量实验
    result, c_result = exp_with_mobile_num()
    record(result, c_result, "exp_with_mobile_num")
    #
    # # 带宽实验
    # result, c_result = exp_with_bandwidth()
    # record(result, c_result, "exp_with_bandwidth")
    #
    # # deadline实验
    # result, c_result = exp_with_deadline()
    # record(result, c_result, "exp_with_deadline")
    # 
    # # 服务器数量实验
    # result, c_result = exp_with_server_num()
    # record(result, c_result, "exp_with_server_num")

    # # 服务率-终端设备数量实验
    # serve_rate, serve_rate_1, serve_rate_2, serve_rate_3 = serve_rate_with_mobile_num()
    # record_serve_rate(serve_rate, "serve_rate_with_mobile_num")
    # record_serve_rate(serve_rate_1, "serve_rate_1_with_mobile_num")
    # record_serve_rate(serve_rate_2, "serve_rate_2_with_mobile_num")
    # record_serve_rate(serve_rate_3, "serve_rate_3_with_mobile_num")

    # # 服务率-服务器数量实验
    # serve_rate, serve_rate_1, serve_rate_2, serve_rate_3 = serve_rate_with_server_num()
    # record_serve_rate(serve_rate, "serve_rate_with_server_num")
    # record_serve_rate(serve_rate_1, "serve_rate_1_with_server_num")
    # record_serve_rate(serve_rate_2, "serve_rate_2_with_server_num")
    # record_serve_rate(serve_rate_3, "serve_rate_3_with_server_num")