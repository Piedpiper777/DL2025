from __future__ import print_function
import torch, gc
import torch.nn as nn

import streamlit as st
import pathlib
import collections
import torchinfo
import torchvision
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as TR
import segmentation_models_pytorch as smp
import albumentations as albu
import scipy.io

import glob
import os

from itertools import islice
import matplotlib.pyplot as plt
from scipy.fft import fft
from tqdm import *
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader,TensorDataset
from scipy.fftpack import fft
sys.path.append('..')

st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="wide",
)
st.set_option('deprecation.showPyplotGlobalUse', False)


def get_streamlit_params():

    # return Dict
    params = collections.defaultdict(int)

    # Choose the DL Library
    library = st.sidebar.selectbox(label="深度学习框架", options=["PyTorch", "TensorFlow"], help="choose you library")
    params["library"] = library

    # Choose the Device 
    if library == "PyTorch":
        available_gpus =  ["cuda:" + str(i) for i in range(torch.cuda.device_count())] + ["cpu"]
        device = st.sidebar.selectbox(label="设备", options=available_gpus)
        params["device"] = device

        # Choose the dataset
        datasets_name = pathlib.Path(f"./user_data/test/rul_datasets").iterdir()
        dataset = st.sidebar.selectbox(label="数据集", options=[n.stem for n in datasets_name if n.is_dir()])
        params["dataset"] = dataset
        
        # Choose 轴承编号
        load_range = pathlib.Path(f"./user_data/test/rul_datasets/{dataset}")
        loads = st.sidebar.selectbox(label="训练轴承", options=os.listdir(load_range))
        params['load'] = loads

        # Choose 轴承编号
        load_range_test = pathlib.Path(f"./user_data/test/rul_datasets/{dataset}")
        loads_test = st.sidebar.selectbox(label="测试轴承", options=os.listdir(load_range_test))
        params['load_test'] = loads_test
        
        # Signal Size
        signal_size = st.sidebar.text_input(label='信号尺寸', value='1200')
        params['signal_size'] = int(signal_size)
        
        # time Size
        time_size = st.sidebar.text_input(label='当前时刻', value='200')
        params['time_size'] = int(time_size)

    if library == "TensorFlow":
        pass

    return params


##### button 1
##### RMS# 均方根值
def RMS(data):
    def get_time_domain_feature(data):
        cols = data.shape[0]
        rms = np.sqrt(np.sum(data ** 2) / cols)  # 均方根值
        features = [rms]
        return np.array(features).T

    def feature_calculation(data):
        '''时域指标'''
        time_domain = []
        for i in range(0, data.shape[0]):
            X_train = data[i]
            time_domain_feature = get_time_domain_feature(X_train)
            time_domain.append(time_domain_feature)
        time_features = np.array(time_domain)

        return time_features

    result = feature_calculation(data)

    plt.figure(figsize=(3.2, 3.6))
    plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    # 绘制RMS特征曲线
    plt.plot(result, 'green', linewidth=1)

    # 设置x轴刻度
    _xtick_labels = np.linspace(0, data.shape[0]-1, num=2, dtype=int)
    plt.xticks(_xtick_labels, ('0', fr'{data.shape[0]}'))

    # 设置刻度字体大小
    plt.tick_params(labelsize=10)

    # 配置中文字体
    font2 = {'family': 'SimHei',
            'weight': 'normal',
            'size': 10}

    # 设置中文标签和标题
    plt.title('均方根特征', font2)
    plt.xlabel('时间 (10秒)', font2)
    plt.ylabel('幅值 (A/V)', font2)

    plt.show()

    st.pyplot()


##### button 1
def baoluopu(shuju):
    SHUJU = []
    for i in range(shuju.shape[0]):
        a1 = shuju[i]
        a = a1.ravel()
        t1 = range(0, len(a))
        yf11 = abs(fft(a)) / len(a)  # 归一化处理
        yf111 = yf11[range(int(len(t1) / 2))]  # 由于对称性，只取一半区间
        SHUJU.append(yf111)
    SHUJU = np.array(SHUJU)
    return SHUJU

def data_label(T_data):
    T_data = baoluopu(T_data)
    print('T_data', T_data.shape)
    T_lable = np.linspace(1,0, T_data.shape[0])
    T_data = np.reshape(T_data, (T_data.shape[0], 1, T_data.shape[1]))
    return T_data, T_lable

def sequence_traindata_loader(T_data, T_lable, T_data4, T_lable4):
    # 转化为tensor数据类型
    x_train = torch.Tensor(T_data)
    y_train = torch.Tensor(T_lable)
    x_val = torch.Tensor(T_data4)
    y_val = torch.Tensor(T_lable4)
    data_train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=100, shuffle=True,num_workers=0)
    data_val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=100, shuffle=False, num_workers=0)
    return data_train_loader, data_val_loader
def sequence_testdata_loader(T_data5, T_lable5, T_data6, T_lable6):
    # 转化为tensor数据类型
    x_test5 = torch.Tensor(T_data5)
    y_test5 = torch.Tensor(T_lable5)
    x_test6 = torch.Tensor(T_data6)
    y_test6 = torch.Tensor(T_lable6)
    data_test_loader5 = DataLoader(TensorDataset(x_test5, y_test5), batch_size=50, shuffle=False, num_workers=0)
    data_test_loader6 = DataLoader(TensorDataset(x_test6, y_test6), batch_size=50, shuffle=False, num_workers=0)
    return data_test_loader5, data_test_loader6


def predict_pytorch(params, predict_button, data_analyze_button):
    # TODO 目前该pipeline只在HUST数据集上完成测试
    # ------------------------
    # Title
    # ------------------------
    # user_name = "test"
    # st.header("剩余使用寿命预测")
    # ------------------------
    # get params config
    # ------------------------

    chosen_fault = "./user_data/test/rul_datasets/{}/{}".format(params["dataset"], params["load"])
    y = scipy.io.loadmat(chosen_fault)['a']
    y = y[:, 0:2560]
    print('y', y.shape)
    
    chosen_fault_test = "./user_data/test/rul_datasets/{}/{}".format(params["dataset"], params["load_test"])
    y_test = scipy.io.loadmat(chosen_fault_test)['a']
    y_test = y_test[:, 0:2560]
    print('y_test', y_test.shape)

    device = params['device']
    device=torch.device(device=device)
    # ------------------------

    # Data Analysis
    # ------------------------
    if data_analyze_button:
        st.header("数据分析")
        col1, col2 = st.columns(2)
        
        with col1:
            aa = 0
            num_batch = 1
            for i in range(aa, num_batch):
                batch_x = y[0:y.shape[0]]
                print('batch_x', batch_x.shape)

                batch_x1 = np.reshape(batch_x, (1,y.shape[0]*y.shape[1]))
                x=[]
                x.append(batch_x1)
                x1 = x[0].tolist()
                d = np.array(x1)
                x2 = d[0].tolist()
                d1 = np.array(x2)

                plt.figure(figsize=(3.2, 3.6))
                plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

                # 配置支持中文的字体
                plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
                plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

                # 绘制信号曲线
                plt.plot(d1, 'darkblue', linewidth=1)

                # 设置x轴刻度
                _xtick_labels = np.linspace(0, y.shape[0]*y.shape[1]-1, num=2, dtype=int)
                plt.xticks(_xtick_labels, ('0', fr'{y.shape[0]}'))

                # 设置刻度字体大小
                plt.tick_params(labelsize=10)

                # 配置中文字体
                font2 = {'family': 'SimHei',
                        'weight': 'normal',
                        'size': 10}

                # 设置中文标签和标题
                plt.xlabel('时间 (10秒)', font2)
                plt.ylabel('幅值 (A/V)', font2)
                plt.title('全寿命周期信号', font2)

                plt.show()
    
                st.pyplot()
            
        file_data = y
        with col2:
            RMS(file_data)
            
    # ------------------------
    # Predict
    # ------------------------
    S_data, S_lable = data_label(y)
    T_data, T_lable =  data_label(y_test)
    
    if predict_button:
        st.header("数据分析")
        col1, col2 = st.columns(2)
        
        with col1:
            aa = 0
            num_batch = 1
            for i in range(aa, num_batch):
                batch_x = y[0:y.shape[0]]
                print('batch_x', batch_x.shape)

                batch_x1 = np.reshape(batch_x, (1,y.shape[0]*y.shape[1]))
                x=[]
                x.append(batch_x1)
                x1 = x[0].tolist()
                d = np.array(x1)
                x2 = d[0].tolist()
                d1 = np.array(x2)

                plt.figure(figsize=(3.2, 3.6))
                plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

                # 配置支持中文的字体
                plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
                plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

                # 绘制信号曲线
                plt.plot(d1, 'darkblue', linewidth=1)

                # 设置x轴刻度
                _xtick_labels = np.linspace(0, y.shape[0]*y.shape[1]-1, num=2, dtype=int)
                plt.xticks(_xtick_labels, ('0', fr'{y.shape[0]}'))

                # 设置刻度字体大小
                plt.tick_params(labelsize=10)

                # 配置中文字体
                font2 = {'family': 'SimHei',
                        'weight': 'normal',
                        'size': 10}

                # 设置中文标签和标题
                plt.xlabel('时间 (10秒)', font2)
                plt.ylabel('幅值 (A/V)', font2)
                plt.title('全寿命周期信号', font2)

                plt.show()
    
                st.pyplot()
            
        file_data = y
        with col2:
            RMS(file_data)
            
          
          
            
        st.header("剩余使用寿命预测")
        col1, col2,  col3 = st.columns(3)
        #加载原始数据并预处理
        data_train_loader,data_val_loader=sequence_traindata_loader(S_data, S_lable, T_data, T_lable)
        
        # 创建网络模型
        model = CNNnet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=50,min_lr=0.001)###设置学习率下降策略，连续50次loss没有下降，学习率为原来的0.1倍，最小值为0.0001
        Losses = nn.MSELoss()

        ###这个是用于更新初始的val_loss的阈值
        epochs=10
        train_loss = []
        train_flood = []
        val_loss = []
        all_epochs = []
        lr_step = []

        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            flood_loss = 0
            model.train()
            for batch_idx, (data, target) in enumerate(data_train_loader):
                total = 0
                data, target = data, target
                optimizer.zero_grad()
                output = model(data)
                output = output.squeeze(1)
                loss = Losses(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total = batch_idx + 1
            epoch_loss = epoch_loss / total


            lr_current = optimizer.param_groups[0]['lr']
            lr_step.append(lr_current)

            print('epoch [{}/{}], loss:{:.4f},  lr:{}'.format(epoch, epochs,epoch_loss,lr_current))  # {:.4f}

            if epoch % 1 == 0:
                train_loss.append(epoch_loss)
                all_epochs.append(epoch)
                
        plt.figure(figsize=(3.2, 3.6))
        plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)
        # 修改字体设置以支持中文显示
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 训练loss查看
        font2 = {'family': 'SimHei',  # 使用支持中文的字体
                'weight': 'normal',
                'size': 10}
        plt.tick_params(labelsize=10)
        # plt.grid(linestyle="--")
        plt.plot(all_epochs, train_loss)
        plt.title("损失值", font2)  # 使用中文标题
        plt.xlabel('迭代次数', font2)  # 使用中文标签
        plt.ylabel('损失', font2)  # 使用中文标签
        # plt.legend(['损失'], loc='upper right')
        plt.show()
        

    
        with col1:
            st.pyplot()
            
        def Last_evaluate2(net, loader):
            net.eval()
            min_val_loss = 0.0
            preds = []
            true = []
            i = 0
            with torch.no_grad():
                for data, target in loader:
                    data, target = data, target
                    output = net(data)
                    output = output.squeeze(1)
                    preds.append(output.detach().cpu().numpy())
                    true.append(target.detach().cpu().numpy())
                    min_val_loss += Losses(output, target).item()
                    i += 1
                preds = np.concatenate(preds)
                true = np.concatenate(true)
                return np.round(preds, 4), np.round(true, 5)
        #RUL预测与可视化
        Sdata_test_loader, Tdata_test_loader = sequence_testdata_loader(S_data, S_lable, T_data, T_lable)
        S_test_preds, S_test_true = Last_evaluate2(model, Sdata_test_loader)
        T_test_preds, T_test_true = Last_evaluate2(model, Tdata_test_loader)

        def myPlot(y_true,y_pred,title):
            mse=mean_squared_error(y_pred,y_true)
            rmse=pow(mse,0.5)
            print('均方根误差RMSE: %.6f' % rmse)
            Mae = mean_absolute_error(y_pred,y_true)
            print('平均绝对误差MAE: %.6f' % Mae)
            Score = r2_score(y_pred,y_true)
            print('得分Score: %.6f' % Score)

            x=range(len(y_true))
            
            plt.figure(figsize=(3.2, 3.6))
            plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)
            # 配置支持中文的字体
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
            plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

            font2 = {'family': 'SimHei',  # 应用中文字体
                    'weight': 'normal',
                    'size': 10}
            plt.tick_params(labelsize=10)
            plt.plot(x, y_true, color='red', label='真实值', linewidth='1')
            plt.plot(x, y_pred, color='blue', label='预测值', linewidth='1')
            # plt.grid('--')
            plt.title(title, font2)  # 标题使用中文字体
            plt.ylabel('剩余使用寿命(%)', font2)  # 完整中文标签
            plt.xlabel('运行时间(*10秒)', font2)  # 完整中文标签
            plt.legend(loc='upper right')  # 显示图例
            plt.show()
        
            return y_true
        
        with col2:
            myPlot(S_test_true, S_test_preds, '训练轴承')
            st.pyplot()
            
        with col3:
            myPlot(T_test_true, T_test_preds, '测试轴承')
            st.pyplot()


        st.header("计算剩余使用寿命")
        col1, col2 = st.columns(2)
        def myPlot_RUL(y_true,y_pred,title):
            dangqian_time = params['time_size']
                        
            mse=mean_squared_error(y_pred,y_true)
            rmse=pow(mse,0.5)
            print('均方根误差RMSE: %.6f' % rmse)
            Mae = mean_absolute_error(y_pred,y_true)
            print('平均绝对误差MAE: %.6f' % Mae)
            Score = r2_score(y_pred,y_true)
            print('得分Score: %.6f' % Score)

            x=range(len(y_true))
            plt.figure(figsize=(3.2, 3.6))
            plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

            # 配置支持中文的字体
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
            plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

            font2 = {'family': 'SimHei',  # 使用中文字体
                    'weight': 'normal',
                    'size': 10}
            plt.tick_params(labelsize=10)

            # 绘制真实值和预测值曲线
            plt.plot(x, y_true, color='red', label='真实剩余寿命', linewidth=1)
            plt.plot(x, y_pred, color='blue', label='预测剩余寿命', linewidth=1)

            # 绘制当前时间竖线
            y = np.linspace(0, 1, dangqian_time)
            x = np.full(dangqian_time, dangqian_time)
            plt.plot(x, y, color='c', label='当前时间', linewidth=1)

            # 设置标题和坐标轴标签
            plt.title(title, font2)
            plt.ylabel('剩余寿命(%)', font2)
            plt.xlabel('运行时间(*10秒)', font2)

            # 显示图例
            plt.legend(loc='best', fontsize=10)
            plt.show()
            

            #当前RUL比例
            yy_true = y_true[dangqian_time]
            #全RUL
            RUL_true = dangqian_time/(1-yy_true)   
            #剩余RUL
            yy_trueRUL = RUL_true-dangqian_time
            
            
            yy_pred = y_pred[dangqian_time]
            yy_predRUL = yy_pred*RUL_true
            
            return RUL_true, yy_trueRUL, yy_predRUL, Mae
        
        with col1:
            myPlot_RUL(T_test_true, T_test_preds, '测试轴承')
            st.pyplot()
        y_true, yy_trueRUL, yy_predRUL, Mae = myPlot_RUL(T_test_true, T_test_preds, '测试轴承')
        '全寿命(*10秒)', y_true
        '真实剩余寿命(*10秒)', yy_trueRUL
        '预测剩余寿命(*10秒)', yy_predRUL
        '预测结果均方误差', Mae
        
        

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
gc.collect()
torch.cuda.empty_cache()
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        '''卷积池化'''
        self.localization = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=(64), stride=(8)),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, kernel_size=(3)),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 64, kernel_size=(3)),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 128, kernel_size=(3)),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))
        '''全连接'''
        self.fc_loc = nn.Sequential(
            nn.Linear(384, 128),
            nn.Dropout(0.5),  # 添加dropout
            nn.Linear(128, 1),
            torch.nn.Sigmoid())
    def forward(self, x):
        # x = x.to(torch.float32)
        x0 = self.localization(x)
        x_flat = x0.view(x0.size(0), -1)
        theta = self.fc_loc(x_flat)
        return theta
    

def predict_tensorflow():
    pass



def main():
    params = get_streamlit_params()
    library = params["library"]

    data_analyze_button = st.sidebar.button(label='数据展示')
    predict_button = st.sidebar.button(label="开始预测")
    
    if library == "PyTorch":
        predict_pytorch(params, predict_button, data_analyze_button)
    elif library == "TensorFlow":
        predict_tensorflow()


if __name__ == '__main__':
    main()