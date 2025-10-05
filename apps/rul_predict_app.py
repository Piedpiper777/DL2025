import streamlit as st
import collections
import numpy as np
import pathlib
import os
from itertools import islice
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import fftpack, signal
import pandas as pd
import pywt
import stockwell
from scipy import stats
import scipy.io
from scipy.optimize import leastsq


st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="wide",
)

def main():
    st.header("数据准备")
    clicked = False
    # create three cols
    # 所选数据集及轴承编号
    col1, col2 = st.columns(2)
    with col1:
        datasets_name = pathlib.Path(f"./user_data/test/rul_datasets").iterdir()
        dataset = st.selectbox(label="数据集", options=[n.stem for n in datasets_name if n.is_dir()])
        params["dataset"] = dataset
        
    with col2:    
        # Choose 轴承编号1
        load_range1 = pathlib.Path(f"./user_data/test/rul_datasets/{dataset}")
        loads1 = st.selectbox(label="轴承编号1", options=os.listdir(load_range1))
        params['load1'] = loads1
        
        # Choose 轴承编号2
        load_range2 = pathlib.Path(f"./user_data/test/rul_datasets/{dataset}")
        loads2 = st.selectbox(label="轴承编号2", options=os.listdir(load_range2))
        params['load2'] = loads2
          
          
    col1, col2 = st.columns([2,2], gap='small')
    # page in col1 - 健康指标类型选择
    with col1:
        st.header("健康指标类型")
        clicked = False
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            button1 = st.button('RMS', use_container_width=True)
            if button1:
                params["button"] = 1  # 保持原有变量名，但在不同区域使用
            button2 = st.button('kurtosis', use_container_width=True)
            if button2:
                params["button"] = 2    
            button3 = st.button('kurtosis_factor', use_container_width=True)
            if button3:
                params["button"] = 3  
            button4 = st.button('variance', use_container_width=True)
            if button4:
                params["button"] = 4  
        with col1_2:
            button5 = st.button('skewness', use_container_width=True)
            if button5:
                params["button"] = 5
            button6 = st.button('crest_factor', use_container_width=True)
            if button6:
                params["button"] = 6
            button7 = st.button('impulse_factor', use_container_width=True)
            if button7:
                params["button"] = 7
            button8 = st.button('p_p_value', use_container_width=True)
            if button8:
                params["button"] = 8
                
        # 将button值存于不同文件，避免覆盖
        s = open("./user_data/preprocessing.txt", 'w') 
        s.write(str(params["button"]))
        s.close()
        
    with col2:
        st.header("被试轴承选择")
        clicked = False
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            button1 = st.button('HI13', use_container_width=True)
            if button1:
                params["button_bearing"] = 1  # 使用不同的键名
            button2 = st.button('HI14', use_container_width=True)
            if button2:
                params["button_bearing"] = 2    
            button3 = st.button('HI15', use_container_width=True)
            if button3:
                params["button_bearing"] = 3  
            button4 = st.button('HI17', use_container_width=True)
            if button4:
                params["button_bearing"] = 4  
        with col1_2:
            button5 = st.button('HI26', use_container_width=True)
            if button5:
                params["button_bearing"] = 5
            button6 = st.button('HI33', use_container_width=True)
            if button6:
                params["button_bearing"] = 6
            button7 = st.button('HI31', use_container_width=True)
            if button7:
                params["button_bearing"] = 7
            button8 = st.button('HI32', use_container_width=True)
            if button8:
                params["button_bearing"] = 8
                
        # 将button值存于不同文件
        s = open("./user_data/preprocessing1.txt", 'w') 
        s.write(str(params["button_bearing"]))
        s.close()
        

    col1, col2 = st.columns([2,2], gap='small')
    params['flag'] = False
    # page in col2
    with col2:
        st.header("参数设置")
        # 同时检查两个参数
        if "button" in params.keys() and "button_bearing" in params.keys():
            # 合并显示逻辑，避免重复表单
            metric_names = {
                1: "RMS", 2: "kurtosis", 3: "kurtosis_factor", 4: "variance",
                5: "skewness", 6: "crest_factor", 7: "impulse_factor", 8: "p_p_value"
            }
            
            bearing_names = {
                1: "HI13", 2: "HI14", 3: "HI15", 4: "HI17",
                5: "HI26", 6: "HI33", 7: "HI31", 8: "HI32"
            }
            
            st.text(f"当前健康指标类型:{metric_names.get(params['button'], '未知')}")
            st.text(f"当前被试轴承:{bearing_names.get(params['button_bearing'], '未知')}")
            
            with st.form("template_input"):
                submit = st.form_submit_button()
                
            if submit:
                params['flag'] = True

    st.set_option('deprecation.showPyplotGlobalUse', False)
     
     
     
    
    col1 = st.container()
    with col1:
        st.header("拟合预测过程")
        if params['flag']:
            preprocessing1(params)
            
    col1 = st.container()
    with col1:
        st.header("两阶段拟合预测过程")
        if params['flag']:
            preprocessing2(params)

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
    
    return result
 
 
##### button 2   
##### kurtosis # 峭度
def kurtosis(data):
    def get_time_domain_feature(data):
        kurtosis = stats.kurtosis(data)  # 峭度
        features = [kurtosis]
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
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(result, 'green', linewidth=1)
    _xtick_labels=np.linspace(0, data.shape[0]-1, num=2, dtype=int)
    plt.xticks(_xtick_labels,('0', fr'{data.shape[0]}'))

    plt.tick_params(labelsize=10)
    font2 = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 10}
    plt.xlabel('Time (10s)', font2)
    plt.ylabel('Amplitude (A/V)', font2)
    plt.show()
    plt.title('kurtosis feature')
    st.pyplot()
    
 
##### button 3    
##### kurtosis_factor# 峭度指标
def kurtosis_factor(data):
    def get_time_domain_feature(data):
        cols = data.shape[0]

        rms = np.sqrt(np.sum(data ** 2) / cols)  # 均方根值
        kurtosis = stats.kurtosis(data)  # 峭度
        kurtosis_factor = kurtosis / (rms ** 4)  # 峭度指标

        features = [kurtosis_factor]
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
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(result, 'green', linewidth=1)
    _xtick_labels=np.linspace(0, data.shape[0]-1, num=2, dtype=int)
    plt.xticks(_xtick_labels,('0', fr'{data.shape[0]}'))

    plt.tick_params(labelsize=10)
    font2 = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 10}
    plt.xlabel('Time (10s)', font2)
    plt.ylabel('Amplitude (A/V)', font2)
    plt.show()
    plt.title('kurtosis_factor feature')
    st.pyplot()

##### button 4
##### variance# 方差
def variance(data):
    def get_time_domain_feature(data):
        variance = np.var(data)  # 方差
        features = [variance]
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
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(result, 'green', linewidth=1)
    _xtick_labels=np.linspace(0, data.shape[0]-1, num=2, dtype=int)
    plt.xticks(_xtick_labels,('0', fr'{data.shape[0]}'))

    plt.tick_params(labelsize=10)
    font2 = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 10}
    plt.xlabel('Time (10s)', font2)
    plt.ylabel('Amplitude (A/V)', font2)
    plt.show()
    plt.title('variance feature')
    st.pyplot()

##### button 5
##### skewness# 偏度
def skewness(data):
    def get_time_domain_feature(data):
        skewness = stats.skew(data)  # 偏度
        features = [skewness]
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
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(result, 'green', linewidth=1)
    _xtick_labels=np.linspace(0, data.shape[0]-1, num=2, dtype=int)
    plt.xticks(_xtick_labels,('0', fr'{data.shape[0]}'))

    plt.tick_params(labelsize=10)
    font2 = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 10}
    plt.xlabel('Time (10s)', font2)
    plt.ylabel('Amplitude (A/V)', font2)
    plt.show()
    plt.title('RMS feature')
    st.pyplot()

##### button 6
##### crest_factor 变换
def crest_factor(data):
    def get_time_domain_feature(data):
        cols = data.shape[0]
        peak_value = np.amax(abs(data))  # 最大绝对值
        rms = np.sqrt(np.sum(data ** 2) / cols)  # 均方根值
        crest_factor = peak_value / rms  # 峰值指标

        features = [crest_factor]
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
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(result, 'green', linewidth=1)
    _xtick_labels=np.linspace(0, data.shape[0]-1, num=2, dtype=int)
    plt.xticks(_xtick_labels,('0', fr'{data.shape[0]}'))

    plt.tick_params(labelsize=10)
    font2 = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 10}
    plt.xlabel('Time (10s)', font2)
    plt.ylabel('Amplitude (A/V)', font2)
    plt.show()
    plt.title('crest_factor feature')
    st.pyplot()

##### button 7
##### impulse_factor# 脉冲指标
def impulse_factor(data):
    def get_time_domain_feature(data):
        peak_value = np.amax(abs(data))  # 最大绝对值
        abs_mean = np.mean(abs(data))  # 绝对平均值
        impulse_factor = peak_value / abs_mean  # 脉冲指标

        features = [impulse_factor]
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
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(result, 'green', linewidth=1)
    _xtick_labels=np.linspace(0, data.shape[0]-1, num=2, dtype=int)
    plt.xticks(_xtick_labels,('0', fr'{data.shape[0]}'))

    plt.tick_params(labelsize=10)
    font2 = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 10}
    plt.xlabel('Time (10s)', font2)
    plt.ylabel('Amplitude (A/V)', font2)
    plt.show()
    plt.title('impulse_factor feature')
    st.pyplot()

##### button 8
##### p_p_value# 峰峰值
def p_p_value(data):
    def get_time_domain_feature(data):
        max_value = np.amax(data)  # 最大值
        min_value = np.amin(data)  # 最小值
        p_p_value = max_value - min_value  # 峰峰值
        features = [p_p_value]
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
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(result, 'green', linewidth=1)
    _xtick_labels=np.linspace(0, data.shape[0]-1, num=2, dtype=int)
    plt.xticks(_xtick_labels,('0', fr'{data.shape[0]}'))

    plt.tick_params(labelsize=10)
    font2 = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 10}
    plt.xlabel('Time (10s)', font2)
    plt.ylabel('Amplitude (A/V)', font2)
    plt.show()
    plt.title('p_p_value feature')
    st.pyplot()
    



def nihe_predict_RUL(data):
    data = data.flatten()
    data = data[0:2746]


    plt.rcParams['font.sans-serif'] = ['SimHei']          # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号

    def func(x, p):
        a, b, c, theta = p
        return a * (x**3) + b * (x**2) + c*x  + theta

    def residuals(p, y, x):
        return y - func(x, p)

    def zuixiaoercheng(data1, data2, x1, x2):
        y1 = data1
        y2 = data2
        p0 = np.random.rand(4)                                      # 第一次猜测的函数拟合参数
        print('p0',p0)

        plsq1 = leastsq(residuals, p0, args=(y1, x1))
        plsq2 = leastsq(residuals, p0, args=(y2, x2))
        print("拟合参数1:", plsq1[0])
        print("拟合参数2:", plsq2[0])

        y_fit1 = func(x1, plsq1[0])
        y_fit2 = func(x2, plsq1[0])
        y_fit3 = func(x2, plsq2[0])
        return y1, y_fit1, y2, y_fit2, y_fit3


    l = len(data)
    # print('l', l)
    zhenshi = 200
    data1 = data[0: l - zhenshi]
    data2 = data[0:l]
    bi = len(data1)/len(data2)
    # print('bi', bi)

    x1 = np.linspace(0, bi, len(data1))
    x2 = np.linspace(0, 1, len(data2))
    # print('x1', len(x1))
    # print('x2', len(x2))

    y1, y1_fit, y2, y2_fit, y3_fit = zuixiaoercheng(data1, data2, x1, x2)
    x1 = x1 * len(data)
    x2 = x2 * len(data)

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    plt.tick_params(labelsize=10)

    # 绘制曲线和散点图
    plt.plot(x2, y2_fit, c='k', label="已知RMS拟合曲线")
    plt.scatter(x2, y2, c='b', s=3, marker='*', label='RMS值')
    plt.scatter(x1, y1, c='y', s=4, marker='o', label='已知RMS值')

    # 绘制阈值线
    yu_zhi = np.full(len(x2), 1.5)
    plt.plot(x2, yu_zhi, '--r', label='阈值', linewidth=1.2)

    # 配置中文字体
    font1 = {'family': 'SimHei',
            'weight': 'normal',
            'size': 10}

    # 设置中文标签和图例
    plt.legend(prop=font1, loc='upper left')
    plt.xlabel('时间 (10秒)', font1)
    plt.ylabel('幅值 (A/V)', font1)
    plt.title('RMS特征趋势与预测', font1)  # 添加标题

    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线，提高可读性
    plt.tight_layout()  # 自动调整布局
    plt.show()
    for k in range(0, len(y2_fit)):
        if np.abs(y2_fit[k]) >= 1.5:
            shouming = k - len(data1)
            print("预测剩余寿命(*10秒)", shouming)
            break
    plt.show()
    plt.title('RMS用于RUL预测', font1)
    # "真实剩余寿命(*10秒)200"
    st.pyplot()


def nihe_predict_RUL13(loc):
    def func(x, p):
        a, b, c, theta = p
        return a * (x**3) + b * (x**2) + c*x  + theta
    def residuals(p, y, x):
        return y - func(x, p)
    def zuixiaoercheng(x2, x3, data1):
        p0 = np.random.rand(4)                                      # 第一次猜测的函数拟合参数
        plsq1 = leastsq(residuals, p0, args=(data1, x2))
        y_fit2 = func(x3, plsq1[0])
        return data1, y_fit2

    y = np.loadtxt(loc, delimiter=",", skiprows=0)
    FPT = 2300
    END = y.shape[0]
    data = y[FPT:END]

    l = len(data)
    print('l', l)
    yuce_dianshu = 30
    data1 = data[0: l - yuce_dianshu]
    data2 = data[0:l]
    bi = len(data1)/len(data2)

    x1 = np.linspace(0, 1, len(data2))
    x2 = np.linspace(0, bi, len(data1))
    x3 = np.linspace(0, 1.1, len(data2)+8)
    y1, y2_fit = zuixiaoercheng(x2, x3, data1)

    plt.figure(figsize=(6.8, 5.8))

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    # 绘制曲线
    plt.plot(x1, data2, c='k', label='未知健康指数')
    plt.plot(x2, y1, c='b', label='已知健康指数')
    plt.plot(x3, y2_fit, label='预测健康指数', linewidth=2, color='y')

    # 绘制阈值线
    yuzhi = np.max(data2)
    print('yuzhi', yuzhi)
    yu_zhi = np.full(len(x3), yuzhi)
    plt.plot(x3, yu_zhi, 'r--', label='失效阈值', linewidth=2)

    # 绘制起始位置垂直线
    start_x = x3[l - yuce_dianshu]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='g',
        linestyle='--',
        linewidth=2,
        label='预测起始位置'
    )

    # 设置x轴刻度
    _xtick_labels = np.linspace(0, 1, num=2)
    plt.xticks(_xtick_labels, (fr'{FPT}', fr'{END}'))

    # 设置刻度字体大小
    plt.tick_params(labelsize=15)

    # 配置中文字体
    font2 = {'family': 'SimHei',
            'weight': 'normal',
            'size': 15}

    # 设置中文标签和图例
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('时间', font2)
    plt.ylabel('健康指数(HI)', font2)
    plt.title('健康指数预测与评估', font2)  # 添加标题

    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线，提高可读性
    plt.tight_layout()  # 自动调整布局
    plt.show()
    for k in range(0, len(y2_fit)):
        if np.abs(y2_fit[k]) >= yuzhi:
            shouming = k - len(y1)
            "预测剩余寿命(*10秒)", shouming
            break
    yuce = shouming + len(y1)
    start_x = x3[yuce]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='k',
        linestyle='--',
        linewidth=2,
        label='Failure location')

    cross_points = []
    for i in range(len(y2_fit) - 1):
        if (y2_fit[i] - yuzhi) * (y2_fit[i+1] - yuzhi) < 0:  # 检查是否跨越阈值
            x_cross = x3[i] + (x3[i+1] - x3[i]) * (yuzhi - y2_fit[i]) / (y2_fit[i+1] - y2_fit[i])
            cross_points.append((x_cross, yuzhi))
    if cross_points:
        for (x_cross, y_cross) in cross_points:
            plt.plot(x_cross, y_cross, 'ro', markersize=8, label='Intersection point')  # 红点标出交点
            ss = FPT+len(x1)*x_cross

            plt.text(x_cross, y_cross, f' {ss:.0f}',
                    fontsize=10, verticalalignment='bottom')
    "真实剩余寿命(*10秒)", yuce_dianshu

    plt.legend(loc='best', fontsize=14)  # 调整图例位置和大小
    plt.show()
    st.pyplot()
    
def nihe_predict_RUL14(loc):
    def func(x, p):
        a, b, c, theta = p
        return a * (x**3) + b * (x**2) + c*x  + theta
    def residuals(p, y, x):
        return y - func(x, p)
    def zuixiaoercheng(x2, x3, data1):
        p0 = np.random.rand(4)                                      # 第一次猜测的函数拟合参数
        plsq1 = leastsq(residuals, p0, args=(data1, x2))
        y_fit2 = func(x3, plsq1[0])
        return data1, y_fit2

    y = np.loadtxt(loc, delimiter=",", skiprows=0)
    FPT = 1084
    END = y.shape[0]
    data = y[FPT:END]

    l = len(data)
    print('l', l)
    yuce_dianshu = 90
    data1 = data[0: l - yuce_dianshu]
    data2 = data[0:l]
    bi = len(data1)/len(data2)

    x1 = np.linspace(0, 1, len(data2))
    x2 = np.linspace(0, bi, len(data1))
    x3 = np.linspace(0, 1.1, len(data2)+35)
    y1, y2_fit = zuixiaoercheng(x2, x3, data1)
    plt.figure(figsize=(6.8, 5.8))

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    # 绘制曲线
    plt.plot(x1, data2, c='k', label='未知健康指数')
    plt.plot(x2, y1, c='b', label='已知健康指数')
    plt.plot(x3, y2_fit, label='预测健康指数', linewidth=2, color='y')

    # 绘制阈值线
    yuzhi = np.max(data2)
    print('yuzhi', yuzhi)
    yu_zhi = np.full(len(x3), yuzhi)
    plt.plot(x3, yu_zhi, 'r--', label='失效阈值', linewidth=2)

    # 绘制起始位置垂直线
    start_x = x3[l - yuce_dianshu]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='g',
        linestyle='--',
        linewidth=2,
        label='预测起始位置'
    )

    # 设置x轴刻度
    _xtick_labels = np.linspace(0, 1, num=2)
    plt.xticks(_xtick_labels, (fr'{FPT}', fr'{END}'))

    # 设置刻度字体大小
    plt.tick_params(labelsize=15)

    # 配置中文字体
    font2 = {'family': 'SimHei',
            'weight': 'normal',
            'size': 15}

    # 设置中文标签和图例
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('时间', font2)
    plt.ylabel('健康指数(HI)', font2)
    plt.title('健康指数预测与评估', font2)  # 添加标题
    for k in range(0, len(y2_fit)):
        if np.abs(y2_fit[k]) >= yuzhi:
            shouming = k - len(y1)
            "预测剩余寿命(*10秒)", shouming
            break
    yuce = shouming + len(y1)
    start_x = x3[yuce]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='k',
        linestyle='--',
        linewidth=2,
        label='Failure location')

    cross_points = []
    for i in range(len(y2_fit) - 1):
        if (y2_fit[i] - yuzhi) * (y2_fit[i+1] - yuzhi) < 0:  # 检查是否跨越阈值
            x_cross = x3[i] + (x3[i+1] - x3[i]) * (yuzhi - y2_fit[i]) / (y2_fit[i+1] - y2_fit[i])
            cross_points.append((x_cross, yuzhi))
    if cross_points:
        for (x_cross, y_cross) in cross_points:
            plt.plot(x_cross, y_cross, 'ro', markersize=8, label='Intersection point')  # 红点标出交点
            ss = FPT+len(x1)*x_cross

            plt.text(x_cross, y_cross, f' {ss:.0f}',
                    fontsize=10, verticalalignment='bottom')
    "真实剩余寿命(*10秒)", yuce_dianshu

    plt.legend(loc='best', fontsize=14)  # 调整图例位置和大小
    plt.show()
    st.pyplot()
    
def nihe_predict_RUL15(loc):
    def func(x, p):
        a, b, c, theta = p
        return a * (x**3) + b * (x**2) + c*x  + theta
    def residuals(p, y, x):
        return y - func(x, p)
    def zuixiaoercheng(x2, x3, data1):
        p0 = np.random.rand(4)                                      # 第一次猜测的函数拟合参数
        plsq1 = leastsq(residuals, p0, args=(data1, x2))
        y_fit2 = func(x3, plsq1[0])
        return data1, y_fit2

    y = np.loadtxt(loc, delimiter=",", skiprows=0)
    FPT = 2409
    END = y.shape[0]
    data = y[FPT:END]

    l = len(data)
    print('l', l)
    yuce_dianshu = 15
    data1 = data[0: l - yuce_dianshu]
    data2 = data[0:l]
    bi = len(data1)/len(data2)

    x1 = np.linspace(0, 1, len(data2))
    x2 = np.linspace(0, bi, len(data1))
    x3 = np.linspace(0, 1, len(data2))
    y1, y2_fit = zuixiaoercheng(x2, x3, data1)
    plt.figure(figsize=(6.8, 5.8))

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    # 绘制曲线
    plt.plot(x1, data2, c='k', label='未知健康指数')
    plt.plot(x2, y1, c='b', label='已知健康指数')
    plt.plot(x3, y2_fit, label='预测健康指数', linewidth=2, color='y')

    # 绘制阈值线
    yuzhi = np.max(data2)
    print('yuzhi', yuzhi)
    yu_zhi = np.full(len(x3), yuzhi)
    plt.plot(x3, yu_zhi, 'r--', label='失效阈值', linewidth=2)

    # 绘制起始位置垂直线
    start_x = x3[l - yuce_dianshu]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='g',
        linestyle='--',
        linewidth=2,
        label='预测起始位置'
    )

    # 设置x轴刻度
    _xtick_labels = np.linspace(0, 1, num=2)
    plt.xticks(_xtick_labels, (fr'{FPT}', fr'{END}'))

    # 设置刻度字体大小
    plt.tick_params(labelsize=15)

    # 配置中文字体
    font2 = {'family': 'SimHei',
            'weight': 'normal',
            'size': 15}

    # 设置中文标签和图例
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('时间', font2)
    plt.ylabel('健康指数(HI)', font2)
    plt.title('健康指数预测与评估', font2)  # 添加标题
    for k in range(0, len(y2_fit)):
        if np.abs(y2_fit[k]) >= yuzhi:
            shouming = k - len(y1)
            "预测剩余寿命(*10秒)", shouming
            break
    yuce = shouming + len(y1)
    start_x = x3[yuce]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='k',
        linestyle='--',
        linewidth=2,
        label='Failure location')

    cross_points = []
    for i in range(len(y2_fit) - 1):
        if (y2_fit[i] - yuzhi) * (y2_fit[i+1] - yuzhi) < 0:  # 检查是否跨越阈值
            x_cross = x3[i] + (x3[i+1] - x3[i]) * (yuzhi - y2_fit[i]) / (y2_fit[i+1] - y2_fit[i])
            cross_points.append((x_cross, yuzhi))
    if cross_points:
        for (x_cross, y_cross) in cross_points:
            plt.plot(x_cross, y_cross, 'ro', markersize=8, label='Intersection point')  # 红点标出交点
            ss = FPT+len(x1)*x_cross

            plt.text(x_cross, y_cross, f' {ss:.0f}',
                    fontsize=10, verticalalignment='bottom')
    "真实剩余寿命(*10秒)", yuce_dianshu

    plt.legend(loc='best', fontsize=14)  # 调整图例位置和大小
    plt.show()
    st.pyplot()
    
def nihe_predict_RUL17(loc):
    def func(x, p):
        a, b, c, theta = p
        return a * (x**3) + b * (x**2) + c*x  + theta
    def residuals(p, y, x):
        return y - func(x, p)
    def zuixiaoercheng(x2, x3, data1):
        p0 = np.random.rand(4)                                      # 第一次猜测的函数拟合参数
        plsq1 = leastsq(residuals, p0, args=(data1, x2))
        y_fit2 = func(x3, plsq1[0])
        return data1, y_fit2

    y = np.loadtxt(loc, delimiter=",", skiprows=0)
    FPT = 2208
    END = y.shape[0]
    data = y[FPT:END]

    l = len(data)
    print('l', l)
    yuce_dianshu = 10
    data1 = data[0: l - yuce_dianshu]
    data2 = data[0:l]
    bi = len(data1)/len(data2)

    x1 = np.linspace(0, 1, len(data2))
    x2 = np.linspace(0, bi, len(data1))
    x3 = np.linspace(0, 1.1, len(data2)+5)
    y1, y2_fit = zuixiaoercheng(x2, x3, data1)
    plt.figure(figsize=(6.8, 5.8))

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    # 绘制曲线
    plt.plot(x1, data2, c='k', label='未知健康指数')
    plt.plot(x2, y1, c='b', label='已知健康指数')
    plt.plot(x3, y2_fit, label='预测健康指数', linewidth=2, color='y')

    # 绘制阈值线
    yuzhi = np.max(data2)
    print('yuzhi', yuzhi)
    yu_zhi = np.full(len(x3), yuzhi)
    plt.plot(x3, yu_zhi, 'r--', label='失效阈值', linewidth=2)

    # 绘制起始位置垂直线
    start_x = x3[l - yuce_dianshu]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='g',
        linestyle='--',
        linewidth=2,
        label='预测起始位置'
    )

    # 设置x轴刻度
    _xtick_labels = np.linspace(0, 1, num=2)
    plt.xticks(_xtick_labels, (fr'{FPT}', fr'{END}'))

    # 设置刻度字体大小
    plt.tick_params(labelsize=15)

    # 配置中文字体
    font2 = {'family': 'SimHei',
            'weight': 'normal',
            'size': 15}

    # 设置中文标签和图例
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('时间', font2)
    plt.ylabel('健康指数(HI)', font2)
    plt.title('健康指数预测与评估', font2)  # 添加标题
    for k in range(0, len(y2_fit)):
        if np.abs(y2_fit[k]) >= yuzhi:
            shouming = k - len(y1)
            "预测剩余寿命(*10秒)", shouming
            break
    yuce = shouming + len(y1)
    start_x = x3[yuce]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='k',
        linestyle='--',
        linewidth=2,
        label='Failure location')

    cross_points = []
    for i in range(len(y2_fit) - 1):
        if (y2_fit[i] - yuzhi) * (y2_fit[i+1] - yuzhi) < 0:  # 检查是否跨越阈值
            x_cross = x3[i] + (x3[i+1] - x3[i]) * (yuzhi - y2_fit[i]) / (y2_fit[i+1] - y2_fit[i])
            cross_points.append((x_cross, yuzhi))
    if cross_points:
        for (x_cross, y_cross) in cross_points:
            plt.plot(x_cross, y_cross, 'ro', markersize=8, label='Intersection point')  # 红点标出交点
            ss = FPT+len(x1)*x_cross

            plt.text(x_cross, y_cross, f' {ss:.0f}',
                    fontsize=10, verticalalignment='bottom')
    "真实剩余寿命(*10秒)", yuce_dianshu

    plt.legend(loc='best', fontsize=14)  # 调整图例位置和大小
    plt.show()
    st.pyplot()
    
def nihe_predict_RUL26(loc):
    def func(x, p):
        a, b, c, theta = p
        return a * (x**3) + b * (x**2) + c*x  + theta
    def residuals(p, y, x):
        return y - func(x, p)
    def zuixiaoercheng(x2, x3, data1):
        p0 = np.random.rand(4)                                      # 第一次猜测的函数拟合参数
        plsq1 = leastsq(residuals, p0, args=(data1, x2))
        y_fit2 = func(x3, plsq1[0])
        return data1, y_fit2

    y = np.loadtxt(loc, delimiter=",", skiprows=0)
    FPT = 686
    END = y.shape[0]
    data = y[FPT:END]

    l = len(data)
    print('l', l)
    yuce_dianshu = 5
    data1 = data[0: l - yuce_dianshu]
    data2 = data[0:l]
    bi = len(data1)/len(data2)

    x1 = np.linspace(0, 1, len(data2))
    x2 = np.linspace(0, bi, len(data1))
    x3 = np.linspace(0, 1, len(data2))
    y1, y2_fit = zuixiaoercheng(x2, x3, data1)
    plt.figure(figsize=(6.8, 5.8))

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    # 绘制曲线
    plt.plot(x1, data2, c='k', label='未知健康指数')
    plt.plot(x2, y1, c='b', label='已知健康指数')
    plt.plot(x3, y2_fit, label='预测健康指数', linewidth=2, color='y')

    # 绘制阈值线
    yuzhi = np.max(data2)
    print('yuzhi', yuzhi)
    yu_zhi = np.full(len(x3), yuzhi)
    plt.plot(x3, yu_zhi, 'r--', label='失效阈值', linewidth=2)

    # 绘制起始位置垂直线
    start_x = x3[l - yuce_dianshu]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='g',
        linestyle='--',
        linewidth=2,
        label='预测起始位置'
    )

    # 设置x轴刻度
    _xtick_labels = np.linspace(0, 1, num=2)
    plt.xticks(_xtick_labels, (fr'{FPT}', fr'{END}'))

    # 设置刻度字体大小
    plt.tick_params(labelsize=15)

    # 配置中文字体
    font2 = {'family': 'SimHei',
            'weight': 'normal',
            'size': 15}

    # 设置中文标签和图例
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('时间', font2)
    plt.ylabel('健康指数(HI)', font2)
    plt.title('健康指数预测与评估', font2)  # 添加标题
    for k in range(0, len(y2_fit)):
        if np.abs(y2_fit[k]) >= yuzhi:
            shouming = k - len(y1)
            "预测剩余寿命(*10秒)", shouming
            break
    yuce = shouming + len(y1)
    start_x = x3[yuce]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='k',
        linestyle='--',
        linewidth=2,
        label='Failure location')

    cross_points = []
    for i in range(len(y2_fit) - 1):
        if (y2_fit[i] - yuzhi) * (y2_fit[i+1] - yuzhi) < 0:  # 检查是否跨越阈值
            x_cross = x3[i] + (x3[i+1] - x3[i]) * (yuzhi - y2_fit[i]) / (y2_fit[i+1] - y2_fit[i])
            cross_points.append((x_cross, yuzhi))
    if cross_points:
        for (x_cross, y_cross) in cross_points:
            plt.plot(x_cross, y_cross, 'ro', markersize=8, label='Intersection point')  # 红点标出交点
            ss = FPT+len(x1)*x_cross

            plt.text(x_cross, y_cross, f' {ss:.0f}',
                    fontsize=10, verticalalignment='bottom')
    "真实剩余寿命(*10秒)", yuce_dianshu

    plt.legend(loc='best', fontsize=14)  # 调整图例位置和大小
    plt.show()
    st.pyplot()
    
def nihe_predict_RUL33(loc):
    def func(x, p):
        a, b, c, theta = p
        return a * (x**3) + b * (x**2) + c*x  + theta
    def residuals(p, y, x):
        return y - func(x, p)
    def zuixiaoercheng(x2, x3, data1):
        p0 = np.random.rand(4)                                      # 第一次猜测的函数拟合参数
        plsq1 = leastsq(residuals, p0, args=(data1, x2))
        y_fit2 = func(x3, plsq1[0])
        return data1, y_fit2

    y = np.loadtxt(loc, delimiter=",", skiprows=0)
    y = y[:423]
    FPT = 315
    END = y.shape[0]
    data = y[FPT:END]

    l = len(data)
    print('l', l)
    yuce_dianshu = 40
    data1 = data[0: l - yuce_dianshu]
    data2 = data[0:l]
    bi = len(data1)/len(data2)

    x1 = np.linspace(0, 1, len(data2))
    x2 = np.linspace(0, bi, len(data1))
    x3 = np.linspace(0, 1, len(data2))
    y1, y2_fit = zuixiaoercheng(x2, x3, data1)
    plt.figure(figsize=(6.8, 5.8))

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    # 绘制曲线
    plt.plot(x1, data2, c='k', label='未知健康指数')
    plt.plot(x2, y1, c='b', label='已知健康指数')
    plt.plot(x3, y2_fit, label='预测健康指数', linewidth=2, color='y')

    # 绘制阈值线
    yuzhi = np.max(data2)
    print('yuzhi', yuzhi)
    yu_zhi = np.full(len(x3), yuzhi)
    plt.plot(x3, yu_zhi, 'r--', label='失效阈值', linewidth=2)

    # 绘制起始位置垂直线
    start_x = x3[l - yuce_dianshu]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='g',
        linestyle='--',
        linewidth=2,
        label='预测起始位置'
    )

    # 设置x轴刻度
    _xtick_labels = np.linspace(0, 1, num=2)
    plt.xticks(_xtick_labels, (fr'{FPT}', fr'{END}'))

    # 设置刻度字体大小
    plt.tick_params(labelsize=15)

    # 配置中文字体
    font2 = {'family': 'SimHei',
            'weight': 'normal',
            'size': 15}

    # 设置中文标签和图例
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('时间', font2)
    plt.ylabel('健康指数(HI)', font2)
    plt.title('健康指数预测与评估', font2)  # 添加标题
    plt.legend(loc='best', fontsize=14)  # 调整图例位置和大小
    plt.show()
    
    for k in range(0, len(y2_fit)):
        if np.abs(y2_fit[k]) >= yuzhi:
            shouming = k - len(y1)
            "预测剩余寿命(*10秒)", shouming
            break
    yuce = shouming + len(y1)
    start_x = x3[yuce]  # 起点的 x 坐标
    plt.plot(
        [start_x, start_x],  # x 坐标相同（垂直线）
        [0, yuzhi],  # y 范围：从数据最小值到失效阈值
        color='k',
        linestyle='--',
        linewidth=2,
        label='Failure location')

    cross_points = []
    for i in range(len(y2_fit) - 1):
        if (y2_fit[i] - yuzhi) * (y2_fit[i+1] - yuzhi) < 0:  # 检查是否跨越阈值
            x_cross = x3[i] + (x3[i+1] - x3[i]) * (yuzhi - y2_fit[i]) / (y2_fit[i+1] - y2_fit[i])
            cross_points.append((x_cross, yuzhi))
    if cross_points:
        for (x_cross, y_cross) in cross_points:
            plt.plot(x_cross, y_cross, 'ro', markersize=8, label='Intersection point')  # 红点标出交点
            ss = FPT+len(x1)*x_cross

            plt.text(x_cross, y_cross, f' {ss:.0f}',
                    fontsize=10, verticalalignment='bottom')
    "真实剩余寿命(*10秒)", yuce_dianshu

    st.pyplot()
       
    
# preprocessing
def preprocessing1(params):
    col1, col2, col3 = st.columns(3)
    chosen_fault = "./user_data/test/rul_datasets/{}/{}".format(params["dataset"], params["load1"])

    y = scipy.io.loadmat(chosen_fault)['a']
    y = y[:, 0:2560]
    print('y', y.shape)
    aa = 0
    num_batch = 1
    for i in range(aa,num_batch):
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
    
    with col1:
        st.pyplot()
        
    file_data = y
    with col2:
        if params["button"] == 1:
            rms = RMS(file_data)
        if params["button"] == 2:
            kur = kurtosis(file_data)
        if params["button"] == 3:
            kurt = kurtosis_factor(file_data)
        if params["button"] == 4:
            var = variance(file_data)
        if params["button"] == 5:
            ske = skewness(file_data)
        if params["button"] == 6:
            cres = crest_factor(file_data)
        if params["button"] == 7:
            imp = impulse_factor(file_data)
        if params["button"] == 8:
            p_p = p_p_value(file_data)
            
    with col3:
        if params["button"] == 1:
            nihe_predict_RUL(rms)
        if params["button"] == 2:
            nihe_predict_RUL(kur)
        if params["button"] == 3:
            nihe_predict_RUL(kurt)
        if params["button"] == 4:
            nihe_predict_RUL(var)
        if params["button"] == 5:
            nihe_predict_RUL(ske)
        if params["button"] == 6:
            nihe_predict_RUL(cres)
        if params["button"] == 7:
            nihe_predict_RUL(imp)
        if params["button"] == 8:
            nihe_predict_RUL(p_p)
        
def preprocessing2(params):
    col1, col2, col3 = st.columns(3)
    chosen_fault1 = "./user_data/test/rul_datasets/{}/{}".format(params["dataset"], params["load1"])
    chosen_fault2 = "./user_data/test/rul_datasets/{}/{}".format(params["dataset"], params["load2"])

    y = scipy.io.loadmat(chosen_fault1)['a']
    y = y[:, 0:2560]
    print('y', y.shape)
    aa = 0
    num_batch = 1
    for i in range(aa,num_batch):
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
    
    with col1:
        st.pyplot()
        
    file_data = y
    with col2:
        if params["button"] == 1:
            rms = RMS(file_data)
        if params["button"] == 2:
            kurtosis(file_data)
        if params["button"] == 3:
            kurtosis_factor(file_data)
        if params["button"] == 4:
            variance(file_data)
        if params["button"] == 5:
            skewness(file_data)
        if params["button"] == 6:
            crest_factor(file_data)
        if params["button"] == 7:
            impulse_factor(file_data)
        if params["button"] == 8:
            p_p_value(file_data)
            
    with col3:
        if params["button_bearing"] == 1:
            nihe_predict_RUL13(chosen_fault2)
        if params["button_bearing"] == 2:
            nihe_predict_RUL14(chosen_fault2)
        if params["button_bearing"] == 3:
            nihe_predict_RUL15(chosen_fault2)
        if params["button_bearing"] == 4:
            nihe_predict_RUL17(chosen_fault2)
        if params["button_bearing"] == 5:
            nihe_predict_RUL26(chosen_fault2)
        if params["button_bearing"] == 6:
            nihe_predict_RUL33(chosen_fault2)
        if params["button_bearing"] == 7:
            nihe_predict_RUL33(chosen_fault2)
        if params["button_bearing"] == 8:
            nihe_predict_RUL33(chosen_fault2)


# 读取外部存储的button值
s = open("./user_data/preprocessing.txt", 'r')  
button = s.read()
s1 = open("./user_data/preprocessing1.txt", 'r')  
button_bearing = s1.read()
s.close()


if __name__ == '__main__':
    # return Dict
    params = collections.defaultdict()
    params["button"] = int(button)
    params["button_bearing"] = int(button_bearing)
    main()
