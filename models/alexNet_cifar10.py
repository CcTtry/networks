# -*- coding: utf-8 -*-
# """AlexNet卷积神经网络训练学习CIFAR10"""
import torch.nn as nn
import torch
import json
import datetime
import time
import contextlib
import numpy as np


from SystemType import isWindows
profiling_path = "/home/ctry/gitReg/dnn_inference/networks/profiling/"
if isWindows():
    profiling_path = "E:\\gitReg\\dnn_inference\\networks\\profiling\\"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_func = nn.CrossEntropyLoss()


# 定义网络模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.input_shape = [1, 3, 32, 32]
        self.maxpool1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75))
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True), self.maxpool1)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),  self.maxpool1)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        self.fc = nn.Sequential(  # self.exit5
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc]
        self.cpu_time_profiler = np.zeros(len(self.layers))
        self.gpu_time_profiler = np.zeros(len(self.layers))


    '''Inference'''
    def forward(self, x, start_layer=0, exit_layer=5,  profiler=None, cpu_mode=False):
        if profiler is None:
            for i in range(start_layer, exit_layer + 1):
                x = self.layers[i](x)
            return x
        else:
            for i in range(start_layer, exit_layer + 1):
                with self.profile_time(name_index=i, cpu_mode=cpu_mode):
                    x = self.layers[i](x)
            return x

    def index_to_name(self, index):
        index_map  = {
            0: 'conv-1',
            1: 'conv-2',
            2: 'conv-3',
            3: 'conv-4',
            4: 'conv-5',
            5: 'fc',
        }
        return index_map.get(index, None)

    @contextlib.contextmanager
    def profile_time(self,
                     name_index,
                     stream=None,
                     end_stream=None,
                     cpu_mode=False,
                     cpu_start=0.0,
                     ):
        """profile time spent by CPU and GPU.
        Useful as a temporary context manager to find sweet spots of code
        suitable for async implementation.
        """
        if cpu_mode:
            try:
                cpu_start = time.time()
                yield
            finally:
                cpu_end = time.time()
                cpu_time = (cpu_end - cpu_start) * 1000

                # msg = f'{self.index_to_name(index=name_index)} cpu_time {cpu_time:.2f} ms '
                # print(msg)
                self.cpu_time_profiler[name_index] += cpu_time
        else:
            stream = stream if stream else torch.cuda.current_stream()
            end_stream = end_stream if end_stream else stream
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream.record_event(start)
            try:
                yield
            finally:
                end_stream.record_event(end)
                end.synchronize()
                gpu_time = start.elapsed_time(end)
                msg = f'{self.index_to_name(index=name_index)} gpu_time {gpu_time:.2f} ms'
                self.gpu_time_profiler[name_index] += gpu_time
                # print(msg)

    def dump_json(self, dict={}, test_data_len=10000, cpu_mode=False, filepath=''):
        '''
        :param dict: 保存的结果
        :param filepath: 保存的文件路径
        :param test_data_len: 测试样本的数量
        :param cpu_mode: 测试的时间是否是CPU
        :return:
        '''
        if cpu_mode:
            filepath = f'{profiling_path}cpu_{self.__class__.__name__}_{str(datetime.datetime.now())[0:10]}_timeProfiler.json'
            for i in range(len(self.layers)):
                dict[self.index_to_name(i)] = self.cpu_time_profiler[i]/test_data_len
        else:
            filepath = f'{profiling_path}gpu_{self.__class__.__name__}_{str(datetime.datetime.now())[0:10]}_timeProfiler.json'
            for i in range(len(self.layers)):
                dict[self.index_to_name(i)] = self.gpu_time_profiler[i]/test_data_len
        with open(filepath, 'w') as f:
            json.dump(dict, f, indent=4)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()








