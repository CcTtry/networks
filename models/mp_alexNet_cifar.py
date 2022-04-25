# -*- coding: utf-8 -*-
"""MP-AlexNet卷积神经网络训练学习CIFAR10"""
import contextlib
import torch
import torch.nn as nn
import numpy as np
import time
import datetime
import json
from utils import Profiler
from SystemType import isWindows
profiling_path = "/home/ctry/gitReg/dnn_inference/networks/profiling/"
if isWindows():
    profiling_path = "E:\\gitReg\\dnn_inference\\networks\\profiling\\"

class MP_AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MP_AlexNet, self).__init__()
        self.input_shape = [1, 3, 32, 32]

        self.exit_param = {
            'pool_size': [3, 3, 3, 3, 3, 3],
            'pool_std': [1, 1, 1, 1, 1, 1],
            'fc_input': [8192, 4096, 1536, 1536, 1024],
        }

        self.maxpool1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75))

        # stem
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True), self.maxpool1)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True), self.maxpool1)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        # branches
        self.exit1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.exit2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.exit3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(1536, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.exit4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(1536, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.exit5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        self.exits_num = 5
        self.exits = [self.exit1, self.exit2, self.exit3, self.exit4, self.exit5]
        self.exits = nn.Sequential(*self.exits)

        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.exit5]
        self.layers = nn.Sequential(*self.layers)

        self.cpu_time_profiler = np.zeros(len(self.layers))
        self.gpu_time_profiler = np.zeros(len(self.layers))
        self.b_cpu_time_profiler = np.zeros(len(self.exits))
        self.b_gpu_time_profiler = np.zeros(len(self.exits))
        self.profiler = Profiler(self)

    def index_to_name(self, index):
        index_map  = {
            0: 'conv-1',
            1: 'conv-2',
            2: 'conv-3',
            3: 'conv-4',
            4: 'conv-5',
            5: 'fc',
            -1: 'exit1',
            -2: 'exit2',
            -3: 'exit3',
            -4: 'exit4',
            -5: 'exit5',
        }
        return index_map.get(index, None)

    def forward(self, x, profiler=None, cpu_mode=False):
        outputs = []
        if profiler is not None:
            for i in range(len(self.layers)):
                # stem_layer_name = 'conv' + str(i)
                with self.profile_time(i, cpu_mode=cpu_mode):
                    x = self.layers[i](x)
                if i <= 4:
                    # print(f'exit_name {exit_name}')
                    # exit_name = 'exit' + str(i)
                    with self.profile_time(i, cpu_mode=cpu_mode, if_branch=True):
                        a = self.exits[i](x)

                    outputs.append(a)

                elif i == 5:
                    pass
        else:
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                if i <= 4:
                    print(f'size: {(x.view(x.size(0), -1)).size()}')
                    a = self.exits[i](x)
                    outputs.append(a)
                elif i == 5:
                    pass
        # print(outputs[:-1])
        return x
    def dump_json(self, dict={}, test_data_len=10000, cpu_mode=False, filepath=''):
        '''
        :param dict: 保存的结果
        :param filepath: 保存的文件路径
        :param test_data_len: 测试样本的数量
        :param cpu_mode: 测试的时间是否是CPU
        :return:
        '''
        if cpu_mode:
            filepath = f'{profiling_path}mp_cpu_{self.__class__.__name__}_{str(datetime.datetime.now())[0:10]}_timeProfiler.json'
            for i in range(len(self.layers)):
                dict[self.index_to_name(i)] = self.cpu_time_profiler[i]/test_data_len
                if i < len(self.exits):
                    dict[self.index_to_name(-(i+1))] = self.b_cpu_time_profiler[i]/test_data_len
        else:
            filepath = f'{profiling_path}mp_gpu_{self.__class__.__name__}_{str(datetime.datetime.now())[0:10]}_timeProfiler.json'
            for i in range(len(self.layers)):
                dict[self.index_to_name(i)] = self.gpu_time_profiler[i]/test_data_len
                if i < len(self.exits):
                    dict[self.index_to_name(-(i+1))] = self.b_gpu_time_profiler[i]/test_data_len
        with open(filepath, 'a+') as f:
            json.dump(dict, f, indent=4)

    @contextlib.contextmanager
    def profile_time(self,
                     name_index,
                     stream=None,
                     end_stream=None,
                     cpu_mode=False,
                     if_branch=False,
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

                msg = f'{self.index_to_name(index=name_index)} cpu_time {cpu_time:.2f} ms '
                print(msg)
                if if_branch:
                    self.b_cpu_time_profiler[name_index] += cpu_time
                else:
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
                if if_branch:
                    self.b_gpu_time_profiler[name_index] += gpu_time
                    msg = f'{self.index_to_name(index=-name_index)} gpu_time {gpu_time:.2f} ms'
                else:
                    self.gpu_time_profiler[name_index] += gpu_time
                    msg = f'{self.index_to_name(index=name_index)} gpu_time {gpu_time:.2f} ms'
                print(msg)
