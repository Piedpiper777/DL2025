import streamlit as st
import os
from datetime import datetime
from ai_report_demo import create_pdf_report, generate_charts, get_model_response, generate_diagnosis_conclusion

st.set_page_config(page_title="智能轴承诊断报告生成", layout="wide")

st.title("智能轴承诊断报告生成系统")

# 获取 user_data 目录下所有 csv 文件
data_dir = "user_data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

selected_file = st.selectbox("选择振动数据CSV文件", csv_files)

report_title = st.text_input("报告标题", "轴承振动诊断报告")
project_name = st.text_input("工程名称", "电机实验台测试项目")
test_date = st.date_input("检测日期", datetime.now().date())
# diagnosis_result = st.text_area("补充诊断结论", "根据数据分析，初步判断轴承内圈存在轻微点蚀，建议进一步检查。")
models = ['deepseek-r1:1.5b']  # 可扩展更多模型
model = st.selectbox("选择模型", models ) # 模型名称或路径

if st.button("生成报告") and selected_file:
    data_file_path = os.path.join(data_dir, selected_file)

    st.info("正在处理数据并生成图表...")
    charts_data, signal_stats = generate_charts(data_file_path)

    ai_analysis = ""
    if charts_data and signal_stats:
        st.info("正在调用AI模型进行分析...")
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
        ai_analysis = get_model_response(analysis_prompt, model)
        st.success("AI分析完成。")
    
    st.info("正在生成AI诊断结论...")
    diagnosis_result = generate_diagnosis_conclusion(ai_analysis, signal_stats, model)
    st.success("AI诊断结论生成完毕。")

    st.info("正在生成PDF报告...")
    try:
        st.info("正在生成...")
        pdf_path, pdf_filename = create_pdf_report(
            report_title, project_name, test_date, data_file_path,
            ai_analysis, diagnosis_result, charts_data
        )
        st.success(f"报告生成完毕！文件路径: {pdf_path}")
        with open(pdf_path, "rb") as f:
            st.download_button("下载PDF报告", f, file_name=pdf_filename)
    except Exception as e:
        st.error(f"报告生成失败: {e}")