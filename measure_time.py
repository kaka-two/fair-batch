import torch
import torchvision
import time

iterations = 100
start = 1
num = 21
size = 224

container_name = ["ResNet34", "ResNet50", "ResNet101", "VGG13", "VGG16", "DenseNet121"]

model = torchvision.models.vgg16(pretrained=False, progress=True)
device = torch.device("cuda")
model.to(device)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
print('[', end ='')
for bat in range(start, num):
      random_input = torch.randn(bat, 3, size, size).to(device)
      # GPU预热
      for _ in range(10):
          _ = model(random_input)

      # 测速
      times = 0     # 存储每轮iteration的时间
      with torch.no_grad():
          for iter in range(iterations):
              starter.record()
              _ = model(random_input)
              ender.record()
              # 同步GPU时间 
              torch.cuda.synchronize()
              curr_time = starter.elapsed_time(ender) # 计算时间
              times += curr_time
              # print(curr_time)
      mean_time = times / iterations
      print('%s' % str(round(mean_time, 2)), end = '')
      if bat != num: print(', ', end = '')
print('],')




# print('[', end ='')
# for bat in range(20, 40):
#   bat += 1
#   torch.cuda.synchronize()
#   x = torch.rand(bat, 3, size, size).cuda()
#   model(x)
#   torch.cuda.synchronize()
#   x = torch.rand(bat, 3, size, size)

#   torch.cuda.synchronize()
#   start_time = time.time() 
#   x = x.cuda()
#   model(x)
#   torch.cuda.synchronize()
#   run_time = time.time()-start_time

#   print('%s' % str(round(run_time/1 * 1000, 2)), end = '')
#   if bat != num: print(', ', end = '')
# print('],')

# """
# This file is used to measure the flop of DNN models.
# before using, "pip3 install thop" should be used.
# """
# [0.969, 4.134, 2.022, 11.604, 1.995, 11.309]
# import torch
# from torchvision import models
# from thop import profile
# import math

# container_name = ["ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13"]
# input_size = [164*164*3, 224*224*3, 164*164*3, 224*224*3, 164*164*3, 224*224*3]

# # update the modelP
# net = models.vgg13(pretrained=False, progress=True)
# size = 224
# inputs = torch.randn(1, 3, size, size)
# flops, params = profile(net, inputs=(inputs, ))
# batch = 1
# inputs = torch.randn(batch, 3, size, size)
# flops, params = profile(net, inputs=(inputs, ))
# # output the flops and the network parameter
# print("-"*100)
# print(batch)
# print("flops: ", round(flops / math.pow(10, 9),3), "G")
# print("params: ", params)


# """
# measured in 2021.10.10 in 622 pc by ubuntu

# model_name = [
#     "AlexNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101",
#     "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19"
# ]
# flops = [
#     714, 1819, 3671, 4111, 7833,
#     11558, 7616, 11320, 15483, 19646
# ]
# """
