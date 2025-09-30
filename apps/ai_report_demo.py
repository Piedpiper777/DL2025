# -*- coding: utf-8 -*-

import os
import io
import urllib.request
import platform
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats
import ollama
import matplotlib.pyplot as plt
import matplotlib

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- 字体设置函数 ---
def download_chinese_font():
    """
    下载并保存中文字体文件
    """
    font_dir = "fonts"
    font_path = os.path.join(font_dir, "NotoSansCJK-Regular.ttc")
    if os.path.exists(font_path):
        return font_path
    os.makedirs(font_dir, exist_ok=True)
    try:
        font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTC/NotoSansCJK-Regular.ttc"
        print("正在下载中文字体文件...")
        urllib.request.urlretrieve(font_url, font_path)
        print(f"中文字体下载成功: {font_path}")
        return font_path
    except Exception as e:
        print(f"字体下载失败: {e}")
        try:
            backup_font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimSun.ttf"
            backup_font_path = os.path.join(font_dir, "SimSun.ttf")
            urllib.request.urlretrieve(backup_font_url, backup_font_path)
            print(f"备用字体下载成功: {backup_font_path}")
            return backup_font_path
        except Exception as e2:
            print(f"备用字体下载也失败: {e2}")
            return None

def find_system_chinese_font():
    """
    查找系统中的中文字体
    """
    system = platform.system()
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simkai.ttf"
        ]
    elif system == "Darwin":
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode MS.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
    else:  # Linux
        font_paths = [
            # "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/SimHei.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
    return None

def setup_pdf_fonts():
    """
    设置PDF中文字体，返回注册的字体名称
    """
    try:
        font_path = find_system_chinese_font()
        if not font_path:
            font_path = download_chinese_font()
        
        if font_path and os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                print(f"PDF中文字体注册成功: {font_path}")
                return 'ChineseFont'
            except Exception as e:
                print(f"字体注册失败: {e}")
                return 'Helvetica'
        else:
            print("未找到可用的中文字体文件，使用默认字体")
            return 'Helvetica'
            
    except Exception as e:
        print(f"PDF字体设置失败: {e}")
        return 'Helvetica'

# --- AI 模型调用函数 ---
def get_model_response(prompt, model):
    """
    发送提示词给本地部署的 deepseek-r1:1.5b 模型
    这里我是在本地部署的模型，微调后改路径即可
    """
    try:
        response = ollama.chat(
            model= model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        return f"发生错误：{e}"

def generate_ai_analysis(signal_stats, model):
    """
    生成AI智能分析报告
    """
    analysis_prompt = f"""
    请分析以下轴承振动信号的特征数据，并给出专业的诊断评语：
    
    信号统计信息：
    - 最大幅值: {signal_stats['max_amplitude']:.4f}
    - 有效值(RMS): {signal_stats['rms']:.4f}
    - 标准差: {signal_stats['std']:.4f}
    - 峰峰值: {signal_stats['peak_to_peak']:.4f}
    - 主频率: {signal_stats['dominant_freq']:.2f} Hz
    - 采样率: {signal_stats['sampling_rate']:.0f} Hz
    - 信号长度: {signal_stats['signal_length']} 个采样点
    
    请从以下几个方面进行分析：
    1. 信号幅值水平评估
    2. 频率特征分析
    3. 可能的故障类型判断
    4. 设备状态评估
    5. 建议的维护措施
    
    请用专业但易懂的语言给出200-300字的分析报告。
    """
    return get_model_response(analysis_prompt, model)

def generate_diagnosis_conclusion(ai_analysis, signal_stats, model):
    """
    基于AI分析报告生成最终诊断结论
    """
    diagnosis_prompt = f"""
    基于以下轴承振动信号的AI分析报告和统计数据，请给出简洁明确的诊断结论：

    AI分析报告：
    {ai_analysis}

    关键数据：
    - RMS值: {signal_stats['rms']:.4f}
    - 主频率: {signal_stats['dominant_freq']:.2f} Hz
    - 峰峰值: {signal_stats['peak_to_peak']:.4f}

    请根据以上信息，给出一个简洁的诊断结论（100字以内），包括：
    1. 轴承当前状态判断（正常/异常/故障）
    2. 如果有问题，指出可能的故障类型和位置
    3. 给出具体的维护建议或紧急程度评估

    格式要求：直接给出结论，不需要重复分析过程。
    """
    return get_model_response(diagnosis_prompt, model)

# --- PDF 报告生成函数 ---
def create_pdf_report(report_title, project_name, test_date, selected_file,
                      ai_analysis, diagnosis_result, charts_data=None):
    """
    生成PDF报告
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_filename = f"report_{timestamp}.pdf"
    pdf_path = os.path.join("reports", pdf_filename)
    os.makedirs("reports", exist_ok=True)
    
    chinese_font = setup_pdf_fonts()
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    
    styles = getSampleStyleSheet()
    
    # 自定义样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1,
        fontName=chinese_font
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12,
        fontName=chinese_font
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        fontName=chinese_font
    )

    
    story = []
    
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 20))
    
    # 基本信息表格
    basic_info_data = [
        ['项目', '内容'],
        ['工程名称', project_name],
        ['检测日期', str(test_date)],
        ['生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['数据文件', selected_file if selected_file else '无']
    ]
    basic_info_table = Table(basic_info_data, colWidths=[3*cm, 10*cm])
    basic_info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(basic_info_table)
    story.append(Spacer(1, 20))
    
    # 诊断结论放在前面，突出显示
    if diagnosis_result:
        story.append(Paragraph("诊断结论", heading_style))
        for para in diagnosis_result.split('\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), normal_style))
        story.append(Spacer(1, 20))
    
    if charts_data:
        story.append(Paragraph("数据分析图表", heading_style))
        chart_titles = ["时域波形图", "频谱图", "时频分析图", "包络谱图"]
        for i, (chart_buffer, title) in enumerate(zip(charts_data, chart_titles)):
            story.append(Paragraph(f"{i+1}. {title}", normal_style))
            img = Image(chart_buffer, width=15*cm, height=7.5*cm)
            story.append(img)
            story.append(Spacer(1, 15))
            if i == 1:
                story.append(PageBreak())
    
    if ai_analysis:
        story.append(Paragraph("AI智能分析", heading_style))
        for para in ai_analysis.split('\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), normal_style))
        story.append(Spacer(1, 15))
    
    story.append(Paragraph("备注", heading_style))
    story.append(Paragraph("本报告由智能轴承诊断系统自动生成，包含AI智能分析和数据可视化图表。诊断结论基于AI模型分析生成，仅供参考。", normal_style))
    
    try:
        doc.build(story)
        print(f"PDF报告生成成功: {pdf_path}")
        return pdf_path, pdf_filename
    except Exception as e:
        print(f"PDF生成过程中出现错误: {e}")
        raise e

# --- 数据处理和绘图函数 ---
def generate_charts(file_path):
    """
    读取CSV数据并生成四个图表，返回图表缓冲列表和统计数据
    """
    try:
        df = pd.read_csv(file_path)
        if df.shape[1] < 2:
            raise ValueError("CSV文件至少需要两列（时间, 信号值）")
            
        time = df.iloc[:, 0].values
        signal = df.iloc[:, 1].values
        
        charts_buffers = []
        
        # 1. 时域波形图
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(time, signal, color='steelblue', linewidth=0.8)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Time Domain Waveform")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        buffer1 = io.BytesIO()
        fig1.savefig(buffer1, format='png', dpi=300, bbox_inches='tight')
        buffer1.seek(0)
        charts_buffers.append(buffer1)
        plt.close(fig1)
        
        # 2. 频谱图
        N = len(signal)
        T = time[1] - time[0]
        fft_vals = np.fft.fft(signal)
        fft_freqs = np.fft.fftfreq(N, T)
        fft_vals = np.abs(fft_vals[:N // 2])
        fft_freqs = fft_freqs[:N // 2]
        fft_vals = fft_vals / (np.max(fft_vals) + 1e-8)
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(fft_freqs, fft_vals, color='darkorange', linewidth=0.8)
        ax2.set_xlim(0, np.max(fft_freqs))
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Normalized Amplitude")
        ax2.set_title("Frequency Spectrum")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        buffer2 = io.BytesIO()
        fig2.savefig(buffer2, format='png', dpi=300, bbox_inches='tight')
        buffer2.seek(0)
        charts_buffers.append(buffer2)
        plt.close(fig2)
        
        # 3. 时频图
        from scipy.signal import stft
        f, t_stft, Zxx = stft(signal, fs=1.0 / T, nperseg=256)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        pcm = ax3.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud', cmap='jet')
        fig3.colorbar(pcm, ax=ax3, label="Amplitude")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Frequency (Hz)")
        ax3.set_title("Time-Frequency Analysis")
        plt.tight_layout()
        buffer3 = io.BytesIO()
        fig3.savefig(buffer3, format='png', dpi=300, bbox_inches='tight')
        buffer3.seek(0)
        charts_buffers.append(buffer3)
        plt.close(fig3)
        
        # 4. 包络谱图
        from scipy.signal import hilbert
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        env_fft_vals = np.fft.fft(envelope)
        env_fft_freqs = np.fft.fftfreq(N, T)
        env_fft_vals = np.abs(env_fft_vals[:N // 2])
        env_fft_freqs = env_fft_freqs[:N // 2]
        env_fft_vals = env_fft_vals / (np.max(env_fft_vals) + 1e-8)
        
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(env_fft_freqs, env_fft_vals, color='seagreen', linewidth=0.8)
        ax4.set_xlim(0, 1500)
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Normalized Amplitude")
        ax4.set_title("Envelope Spectrum")
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        buffer4 = io.BytesIO()
        fig4.savefig(buffer4, format='png', dpi=300, bbox_inches='tight')
        buffer4.seek(0)
        charts_buffers.append(buffer4)
        plt.close(fig4)
        
        # 计算统计数据
        signal_stats = {
            'max_amplitude': np.max(np.abs(signal)),
            'rms': np.sqrt(np.mean(signal**2)),
            'std': np.std(signal),
            'peak_to_peak': np.max(signal) - np.min(signal),
            'dominant_freq': fft_freqs[np.argmax(fft_vals)],
            'signal_length': len(signal),
            'sampling_rate': 1.0 / T
        }
        
        return charts_buffers, signal_stats

    except Exception as e:
        print(f"数据处理或图表生成失败: {e}")
        return None, None

# --- 主函数 ---
def main():
    """
    命令行驱动的报告生成主程序
    """
    print("--- 智能轴承诊断报告生成器 ---")
    
    # 示例数据文件，请确保该文件存在
    data_file_path = "example.csv"
   
    # 用户可自定义的报告信息
    report_title = "轴承振动诊断报告"
    project_name = "电机实验台测试项目"
    test_date = datetime.now().date()
    
    print("\n正在处理数据并生成图表...")
    charts_data, signal_stats = generate_charts(data_file_path)
    
    if not charts_data or not signal_stats:
        print("❌ 数据处理失败，无法生成报告")
        return
    
    print("\n正在调用AI模型进行智能分析...")
    ai_analysis = generate_ai_analysis(signal_stats)
    print("✅ AI智能分析完成")
    
    print("\n正在生成AI诊断结论...")
    diagnosis_result = generate_diagnosis_conclusion(ai_analysis, signal_stats)
    print("✅ AI诊断结论生成完成")
    
    print("\n正在生成PDF报告...")
    try:
        pdf_path, pdf_filename = create_pdf_report(
            report_title, project_name, test_date, data_file_path,
            ai_analysis, diagnosis_result, charts_data
        )
        print(f"\n✅ 报告生成完毕！文件路径: {pdf_path}")
        print(f"📄 报告文件名: {pdf_filename}")
        
        # 显示诊断结论预览
        print(f"\n🔍 诊断结论预览：")
        print("-" * 50)
        print(diagnosis_result)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")

if __name__ == "__main__":
    # 初始化matplotlib字体设置
    try:
        if platform.system() == "Windows":
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        elif platform.system() == "Darwin":
            matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        else:
            matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except:
        print("无法自动设置matplotlib中文字体，可能影响图表显示。")
        
    main()