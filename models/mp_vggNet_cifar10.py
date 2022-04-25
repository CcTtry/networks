# -*- coding: utf-8 -*-
"""MP_VGGNet卷积神经网络训练学习CIFAR10"""
import math
import numpy as np
import torch
import time
import contextlib
import torch.nn as nn
import json
import datetime

from SystemType import isWindows
profiling_path = "/home/ctry/gitReg/dnn_inference/networks/profiling/"
if isWindows():
    profiling_path = "E:\\gitReg\\dnn_inference\\networks\\profiling\\"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg = {
    'VGG16-1': [64, 64, 'M'],
    'VGG16-2': [128, 128, 'M'],
    'VGG16-3': [256, 256, 256, 'M'],
    'VGG16-4': [512, 512, 512, 'M'],
    'VGG16-5': [512, 512, 512, 'M'],
    'VGG16-AvgPool': ['A'],
}


class MP_VGGNet(nn.Module):  # 模型需继承nn.Module
    def __init__(self, num_classes=10):
        super(MP_VGGNet, self).__init__()
        self.VGG16_1 = self._make_layers(3, cfg['VGG16-1'])   # "conv1+BN+relu+MaxPool"
        self.VGG16_2 = self._make_layers(64, cfg['VGG16-2'])  # "conv2+BN+relu+MaxPool"
        self.VGG16_3 = self._make_layers(128, cfg['VGG16-3'])  # "conv3+BN+relu+MaxPool"
        self.VGG16_4 = self._make_layers(256, cfg['VGG16-4'])  # "conv4+BN+relu+MaxPool"
        self.VGG16_5 = self._make_layers(512, cfg['VGG16-5'])  # "conv5+BN+relu+MaxPool"
        self.avgpool = self._make_branch(cfg['VGG16-AvgPool'])  # "avgPool"
        self.fc1 = nn.Sequential(nn.Dropout(), nn.Linear(512, 512), nn.ReLU(inplace=True), )  # FC1
        self.fc2 = nn.Sequential(nn.Dropout(), nn.Linear(512, 256), nn.ReLU(inplace=True), )  # FC2
        self.fc3 = nn.Linear(256, num_classes)  # branch 3

        self.exit1 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(16384, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.exit2 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(8192, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.exit3 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(4096, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        self.exit4 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.exit5 = nn.Sequential(
            self.avgpool,
            nn.Flatten(),
            self.fc1,
            self.fc2,
            self.fc3
        )

        self.layers = [self.VGG16_1, self.VGG16_2, self.VGG16_3, self.VGG16_4, self.VGG16_5, self.avgpool, self.fc1, self.fc2, self.fc3 ]
        self.layers = nn.Sequential(*self.layers)

        self.exits = [self.exit1, self.exit2, self.exit3, self.exit4, self.exit5]
        self.exits = nn.Sequential(*self.exits)

        self.cpu_time_profiler = np.zeros(len(self.layers))
        self.gpu_time_profiler = np.zeros(len(self.layers))
        self.b_cpu_time_profiler = np.zeros(len(self.exits))
        self.b_gpu_time_profiler = np.zeros(len(self.exits))


    '''Inference'''
    def forward(self, x, start_layer=0, exit_layer=8,  profiler=None, cpu_mode=False):
        outputs = []
        if profiler is not None:
            for i in range(start_layer, exit_layer + 1):
                with self.profile_time(i, cpu_mode=cpu_mode):
                    x = self.layers[i](x)
                if i <= 4:
                    # print(f'exit_name {exit_name}')
                    # exit_name = 'exit' + str(i)
                    with self.profile_time(i, cpu_mode=cpu_mode, if_branch=True):
                        a = self.exits[i](x)
                elif i == 5:
                    with self.profile_time(i, cpu_mode=cpu_mode, if_branch=False):
                        x = x.view(x.size(0), -1)
            return x
        else:
            for i in range(start_layer, exit_layer + 1):
                x = self.layers[i](x)
                if i <= 4:
                    print(f'size: {(x.view(x.size(0), -1)).size()}')
                    a = self.exits[i](x)
                    outputs.append(a)
                elif i == 5:
                    x = x.view(x.size(0), -1)
            return x


    def _make_layers(self, in_channels, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           # nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


    def _make_branch(self, brch):
        branches = []  # flag = False
        for x in brch:
            if x == 'M':
                branches += [nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]
            elif x == 'A':
                branches += [nn.AvgPool2d(kernel_size=1, stride=1)]
            else:
                branches += [nn.Conv2d(x, x, kernel_size=3, padding=1),
                             nn.BatchNorm2d(x),
                             nn.ReLU(inplace=True)]
        return nn.Sequential(*branches)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def index_to_name(self, index):
        index_map  = {
            0: 'VGG16-1',
            1: 'VGG16-2',
            2: 'VGG16-3',
            3: 'VGG16-4',
            4: 'VGG16-5',
            5: 'avgpool',
            6: 'fc1',
            7: 'fc2',
            8: 'fc3',
            -1: 'exit1',
            -2: 'exit2',
            -3: 'exit3',
            -4: 'exit4',
            -5: 'exit5',

        }
        return index_map.get(index, None)

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

                # print(msg)
                if if_branch:
                    # msg = f'{self.index_to_name(index=-(name_index+1))} cpu_time {cpu_time:.2f} ms '
                    self.b_cpu_time_profiler[name_index] += cpu_time
                else:
                    # msg = f'{self.index_to_name(index=name_index)} cpu_time {cpu_time:.2f} ms '
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
                    msg = f'{self.index_to_name(index=-(name_index+1))} gpu_time {gpu_time:.2f} ms'
                    print(msg)
                else:
                    self.gpu_time_profiler[name_index] += gpu_time
                    msg = f'{self.index_to_name(index=name_index)} gpu_time {gpu_time:.2f} ms'


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
                dict[self.index_to_name(i)] = self.cpu_time_profiler[i] / test_data_len
                if i < len(self.exits):
                    dict[self.index_to_name(-(i + 1))] = self.b_cpu_time_profiler[i] / test_data_len
        else:
            filepath = f'{profiling_path}mp_gpu_{self.__class__.__name__}_{str(datetime.datetime.now())[0:10]}_timeProfiler.json'
            for i in range(len(self.layers)):
                dict[self.index_to_name(i)] = self.gpu_time_profiler[i] / test_data_len
                if i < len(self.exits):
                    dict[self.index_to_name(-(i + 1))] = self.b_gpu_time_profiler[i] / test_data_len
        with open(filepath, 'a+') as f:
            json.dump(dict, f, indent=4)



