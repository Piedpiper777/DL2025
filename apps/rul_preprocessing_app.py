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
        
        # Choose 轴承编号
        load_range = pathlib.Path(f"./user_data/test/rul_datasets/{dataset}")
        loads = st.selectbox(label="轴承编号", options=os.listdir(load_range))
        params['load'] = loads
        
    with col2:    

        signal_size = st.number_input(label='信号长度', value=2560)
        params['signal_size'] = signal_size
    
    
    
    col1, col2 = st.columns([2,2], gap='small')
    # page in col1
    with col1:
        st.header("健康指标类型")
        clicked = False
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            button1 = st.button('RMS', use_container_width=True)
            if button1:
                params["button"] = 1 
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
                
        # 将button值及时存于外部文件，以防刷新
        s = open("./user_data/preprocessing.txt", 'w') 
        s.write(str(params["button"]))
        s.close()

        
    params['flag'] = False
    # page in col2
    with col2:
        st.header("参数设置")
        if "button" in params.keys():
            if params["button"] == 1:
                st.text("失效阈值计算:RMS")
                with st.form("template_input"):
                    esLim_yuzhi = st.number_input("esLim_lower", value=3.0)
                    esLim_num = st.number_input("esLim_num", value=500)
                    esLim_freq = st.number_input("esLim_freq", value=20000)
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["esLim_yuzhi"] = esLim_yuzhi
                    params["esLim_num"] = esLim_num
                    params["esLim_freq"] = esLim_freq
                    
            if params["button"] == 2:
                st.text("当前健康指标类型:kurtosis")
                with st.form("template_input"):
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    
            if params["button"] == 3:
                st.text("当前健康指标类型:kurtosis_factor")
                with st.form("template_input"):
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    
            if params["button"] == 4:
                st.text("当前健康指标类型:variance")
                with st.form("template_input"):
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    
            if params["button"] == 5:
                st.text("当前健康指标类型:skewness")
                with st.form("template_input"):
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    
            if params["button"] == 6:
                st.text("当前健康指标类型:crest_factor")
                with st.form("template_input"):
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    
            if params["button"] == 7:
                st.text("当前健康指标类型:impulse_factor")
                with st.form("template_input"):
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    
            if params["button"] == 8:
                st.text("当前健康指标类型:p_p_value")
                with st.form("template_input"):
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
     
    st.set_option('deprecation.showPyplotGlobalUse', False)   
     
     
     
    # col1 = st.container()
    # # page in col3
    # with col1:
    #     st.header("健康监测过程")
    #     if params['flag']:
    #         preprocessing_1(params)


    col1 = st.container()
    # page in col3
    with col1:
        st.header("异常监测结果")
        if params['flag']:
            preprocessing_2(params)


    col1 = st.container()
    # page in col3
    with col1:
        st.header("早期故障诊断")
        if params['flag']:
            preprocessing_3_1(params)
            preprocessing_3_2(params)
            
            
sisi, zizi = 2.2, 1.3
##### button 1
##### RMS# 均方根值
def RMS1(data):
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

    plt.figure(figsize=(sisi, zizi))
    plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

    # 配置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 绘制曲线
    plt.plot(result, 'green', linewidth=1)

    # 设置x轴刻度
    _xtick_labels = np.linspace(0, data.shape[0]-1, num=2, dtype=int)
    plt.xticks(_xtick_labels, ('0', fr'{data.shape[0]}'))

    # 设置刻度字体大小
    plt.tick_params(labelsize=10)

    # 中文字体配置
    font2 = {'family': 'SimHei',
            'weight': 'normal',
            'size': 10}

    # 设置标签和标题（注意标题需放在show()之前）
    plt.xlabel('时间 (10秒)', font2)
    plt.ylabel('幅值 (A/V)', font2)
    plt.title('均方根特征', font2)  # 标题移至plt.show()之前才能显示

    plt.show()  # 显示图像
    st.pyplot()
 
def RMS2(data, esLim_yuzhi, esLim_num):
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

    HI = feature_calculation(data)

    plt.figure(figsize=(sisi, zizi))
    plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    # 绘制原始RMS曲线和滤波后曲线
    plt.plot(HI, label='均方根值', linewidth=1, markersize=1, color='g')

    HII = HI.flatten()
    from scipy.signal import savgol_filter
    sgHI = savgol_filter(HII, 11, 1, mode='nearest')
    plt.plot(sgHI, color='b', label='平滑后的均方根值')

    def fpt(HI, esLim_num, sgHI, esLim_yuzhi):
        # 选l个点出来求阈值
        x = HI[0:esLim_num]
        arr_mean = np.mean(x)
        arr_std = np.std(x, ddof=1)
        yu_zhi1 = arr_mean + esLim_yuzhi * arr_std
        print('yu_zhi1', yu_zhi1)
        yu_zhi = np.full(len(HI), yu_zhi1)
        t = range(0, len(yu_zhi))
        plt.plot(t, yu_zhi, '--r', label='阈值', linewidth=1.5)

        # 设置刻度字体大小
        plt.tick_params(labelsize=5)
        
        # 配置中文字体
        font1 = {'family': 'SimHei',
                'weight': 'normal',
                'size': 5}  # 增大字体大小，从5调整为10
        
        # 设置中文标签和图例
        plt.legend(prop=font1, loc='best')
        plt.xlabel('时间 (10秒)', font1)
        plt.ylabel('幅值 (A/V)', font1)
        plt.title('RMS特征与阈值分析', font1)  # 添加标题
        
        # plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线，提高可读性


        # print(sgHI)
        # Initialize shouming to None (in case no point meets the condition)
        shouming = None
        # Detect early degradation point
        for k in range(len(sgHI)):
            # print(f"HII[{k}] = {sgHI[k]}")
            if sgHI[k] >= yu_zhi1:
                shouming = k
                "发现退化点:", shouming
                break
                
        # Handle case where no point meets the condition
        if shouming is None:
            "未发现早期退化点"
            
    fpt(HI, esLim_num, sgHI, esLim_yuzhi)
    
    st.pyplot()
    
    
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

    # 设置中文标签和标题（修复了标题位置）
    plt.title('均方根特征', font2)  # 移到show()之前
    plt.xlabel('时间 (10秒)', font2)
    plt.ylabel('幅值 (A/V)', font2)

    plt.show()  # 显示图像
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

    plt.figure(figsize=(sisi, zizi))
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
    
    
    
# preprocessing
def preprocessing_1(params):
    col1, col2 = st.columns(2)
    chosen_fault = "./user_data/test/rul_datasets/{}/{}".format(params["dataset"], params["load"])

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

        plt.figure(figsize=(sisi, zizi))
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
        plt.tick_params(labelsize=5)

        # 配置中文字体
        font2 = {'family': 'SimHei',
                        'weight': 'normal',
                        'size': 5}

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
            RMS1(file_data)
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


# preprocessing
def preprocessing_2(params):
    col1, col2 = st.columns(2)
    chosen_fault = "./user_data/test/rul_datasets/{}/{}".format(params["dataset"], params["load"])

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

        plt.figure(figsize=(sisi, zizi))
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
        plt.tick_params(labelsize=5)

        # 配置中文字体
        font2 = {'family': 'SimHei',
                        'weight': 'normal',
                        'size': 5}

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
            RMS2(file_data, params["esLim_yuzhi"], params["esLim_num"])
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
            

def preprocessing_3_1(params):
    col1, col2 = st.columns(2)
    chosen_fault = "./user_data/test/rul_datasets/{}/{}".format(params["dataset"], params["load"])

    y = scipy.io.loadmat(chosen_fault)['a']
    y = y[:, 0:20000]
    print('y', y.shape)
    
    a111 = y[500]
    plt.figure(figsize=(sisi, zizi))
    plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    plt.tick_params(labelsize=5)
    plt.plot(a111, color='g', linewidth=0.5)

    # 配置中文字体
    font = {'family': 'SimHei',
            'weight': 'normal',
            'size': 5}

    # 设置中文标签
    plt.xlabel('采样点', font)
    plt.ylabel('幅值(g)', font)
    plt.title('信号幅值曲线', font)  # 添加标题

    plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线，提高可读性
    plt.tight_layout()  # 自动调整布局
    plt.show()
    
    with col1:
        st.pyplot()
        
    file_data = y
    with col2:
        if params["button"] == 1:
            from scipy import fftpack
            from scipy.fftpack import fft,ifft
            hx = fftpack.hilbert(a111)  # 希尔伯特变换后的
            data = np.sqrt(hx ** 2 + a111 ** 2)  # 包络谱
            x_mean = np.mean(data)
            a = data - x_mean  # 去直流分量
    
            Fs = params["esLim_freq"]  # 采样频率48k
            n = len(a)  # length of the signal      #样点个数243938
            k = np.arange(n)
            T = n / Fs
            frq = k / T  # two sides frequency range                    取一半
            frq1 = frq[range(int(n / 2))]  #one side frequency range    取一半

            t1 = range(0, len(a))

            y1 = fft(a)  # 快速傅里叶变换
            yf1 = abs(fft(a))  # 取绝对值
            yf11 = abs(fft(a)) / len(a)  # 归一化处理
            yf111 = yf11[range(int(len(t1) / 2))]  # 由于对称性，只取一半区间
    
            plt.figure(figsize=(sisi, zizi))
            plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

            # 配置支持中文的字体
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
            plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

            plt.tick_params(labelsize=5)
            plt.plot(frq1[1:1500], yf111[1:1500], color='g', linewidth=0.5)

            # 配置中文字体
            font = {'family': 'SimHei',
                    'weight': 'normal',
                    'size': 5}

            # 设置中文标签和标题
            plt.xlabel('频率(Hz)', font)
            plt.ylabel('幅值(g)', font)
            plt.title('第530分钟', font)  # 将标题翻译为中文

            plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线，提高可读性
            plt.tight_layout()  # 自动调整布局
            plt.show()
            st.pyplot()
            
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

def preprocessing_3_2(params):
    col1, col2 = st.columns(2)
    chosen_fault = "./user_data/test/rul_datasets/{}/{}".format(params["dataset"], params["load"])

    y = scipy.io.loadmat(chosen_fault)['a']
    y = y[:, 0:20000]
    print('y', y.shape)
    
    a111 = y[837]
    plt.figure(figsize=(sisi, zizi))
    plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

    # 配置支持中文的字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

    plt.tick_params(labelsize=5)
    plt.plot(a111, color='g', linewidth=0.5)

    # 配置中文字体
    font = {'family': 'SimHei',
            'weight': 'normal',
            'size': 5}

    # 设置中文标签
    plt.xlabel('采样点', font)
    plt.ylabel('幅值(g)', font)
    plt.title('信号幅值曲线', font)  # 添加标题

    plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线，提高可读性
    plt.tight_layout()  # 自动调整布局
    plt.show()
    
    with col1:
        st.pyplot()
        
        
    file_data = y
    with col2:
        if params["button"] == 1:
            from scipy import fftpack
            from scipy.fftpack import fft,ifft
            hx = fftpack.hilbert(a111)  # 希尔伯特变换后的
            data = np.sqrt(hx ** 2 + a111 ** 2)  # 包络谱
            x_mean = np.mean(data)
            a = data - x_mean  # 去直流分量
    
            Fs = params["esLim_freq"]  # 采样频率48k
            n = len(a)  # length of the signal      #样点个数243938
            k = np.arange(n)
            T = n / Fs
            frq = k / T  # two sides frequency range                    取一半
            frq1 = frq[range(int(n / 2))]  #one side frequency range    取一半

            t1 = range(0, len(a))

            y1 = fft(a)  # 快速傅里叶变换
            yf1 = abs(fft(a))  # 取绝对值
            yf11 = abs(fft(a)) / len(a)  # 归一化处理
            yf111 = yf11[range(int(len(t1) / 2))]  # 由于对称性，只取一半区间
    
            plt.figure(figsize=(sisi, zizi))
            plt.gcf().subplots_adjust(left=0.15, top=0.91, bottom=0.19)

            # 配置支持中文的字体
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
            plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

            plt.tick_params(labelsize=5)
            plt.plot(frq1[1:1500], yf111[1:1500], color='g', linewidth=0.5)


            # l1 = max(yf111[235:240])
            # l2 = max(yf111[445:470])
            # l3 = max(yf111[685:700])

            # 计算各区间最大值及其在yf111中的索引
            l1 = max(yf111[225:240])
            l1_index = np.argmax(yf111[225:240]) + 225
            l2 = max(yf111[445:470])
            l2_index = np.argmax(yf111[445:470]) + 445
            l3 = max(yf111[685:700])
            l3_index = np.argmax(yf111[685:700]) + 685
            '发现1倍频:', np.argmax(yf111[225:240]) + 225
            '发现2倍频:', np.argmax(yf111[445:470]) + 445
            '发现3倍频:', np.argmax(yf111[685:700]) + 685
            
            # 在图上标记所有最大值点
            plt.scatter(frq1[l1_index], l1, color='red', s=20, label=f'峰值1: {l1:.4f}')
            plt.scatter(frq1[l2_index], l2, color='blue', s=20, label=f'峰值2: {l2:.4f}')
            plt.scatter(frq1[l3_index], l3, color='purple', s=20, label=f'峰值3: {l3:.4f}')

            # 在索引位置的顶部添加标记点（增加一定偏移量）
            offset = max(yf111[1:1500]) * 0.05  # 偏移量为最大值的5%
            plt.scatter(frq1[l1_index], l1 + offset, color='red', s=30, marker='^')
            plt.scatter(frq1[l2_index], l2 + offset, color='blue', s=30, marker='^')
            plt.scatter(frq1[l3_index], l3 + offset, color='purple', s=30, marker='^')


            plt.legend(fontsize=5)
            # plt.xlabel('频率(Hz)', fontsize=5)
            # plt.ylabel('幅值', fontsize=5)
            # plt.title('频谱分析', fontsize=5)
            # plt.grid(True, linestyle='--', alpha=0.5)
            # plt.tight_layout()
            # plt.show()

            # 配置中文字体
            font = {'family': 'SimHei',
                    'weight': 'normal',
                    'size': 5}

            # 设置中文标签和标题
            plt.xlabel('频率(Hz)', font)
            plt.ylabel('幅值(g)', font)
            plt.title('第537分钟', font)  # 将标题翻译为中文

            # plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线，提高可读性
            plt.tight_layout()  # 自动调整布局
            plt.show()
            st.pyplot()
            
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
                      
            
# 读取外部存储的button值
s = open("./user_data/preprocessing.txt", 'r')  
button = s.read()
s.close()


if __name__ == '__main__':
    # return Dict
    params = collections.defaultdict()
    params["button"] = int(button)
    main()
