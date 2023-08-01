# """
# This file is used to measure the inference time of DNN models.
# """
import torch
import torchvision
import time



loop = 20
num = 30
size = 112
bat = 15

# container_name = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "DenseNet121", "DenseNet161"]
# input_size = [224*224*3, 224*224*3, 112*112*3, 112*112*3, 112*112*3, 224*224*3]
model = torchvision.models.densenet121(pretrained=False, progress=True)
model.cuda()


# ##### 获取某批次的处理时间
# # 预热
# x = torch.rand(bat, 3, size, size).cuda()
# model(x)
# x = torch.rand(bat, 3, size, size)

# start_time = time.time()
# for _ in range(loop): 
#   x = x.cuda()
#   model(x)
# run_time = time.time()-start_time

# print('%s' % str(round(run_time/loop * 1000, 2)))

##### 获取某批次范围的处理时间
print('[', end ='')
for bat in range(0, num):
  bat += 1
  # 预热
  x = torch.rand(bat, 3, size, size).cuda()
  model(x)
  x = torch.rand(bat, 3, size, size)

  start_time = time.time()
  for _ in range(loop): 
    x = x.cuda()
    model(x)
  run_time = time.time()-start_time

  print('%s' % str(round(run_time/loop * 1000, 2)), end = '')
  if bat != num: print(', ', end = '')
print('],')


