import streamlit as st
import sqlite3
import os
from datetime import datetime
import cv2
import numpy as np


# 初始化数据库连接
def init_db():
    conn = sqlite3.connect('data_storage.db')
    c = conn.cursor()
    # 创建缺陷检测表
    c.execute('''CREATE TABLE IF NOT EXISTS defect_detection (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 image_path TEXT,
                 defect_type TEXT,
                 location TEXT,
                 severity INTEGER,
                 detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')
    # 创建缺陷分类表
    c.execute('''CREATE TABLE IF NOT EXISTS defect_classification (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 defect_id INTEGER,
                 classification_result TEXT,
                 confidence REAL,
                 classification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY (defect_id) REFERENCES defect_detection (id)
                 )''')
    # 创建故障诊断表
    c.execute('''CREATE TABLE IF NOT EXISTS fault_diagnosis (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 device_id TEXT,
                 fault_type TEXT,
                 fault_location TEXT,
                 diagnosis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')
    # 创建原始图像表
    c.execute('''CREATE TABLE IF NOT EXISTS original_images (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 image_path TEXT,
                 upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')
    # 创建中间结果表（示例为缺陷检测中间结果）
    c.execute('''CREATE TABLE IF NOT EXISTS intermediate_results (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 result_type TEXT,
                 result_data TEXT,
                 result_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')
    # 创建最终结果表（综合表）
    c.execute('''CREATE TABLE IF NOT EXISTS final_results (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 result_type TEXT,
                 result_data TEXT,
                 result_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')
    conn.commit()
    conn.close()


# 存储原始图像
def store_original_image(image):
    if image is not None:
        # 保存图像文件
        image_path = os.path.join("images", f"original_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        cv2.imwrite(image_path, image)
        conn = sqlite3.connect('data_storage.db')
        c = conn.cursor()
        c.execute("INSERT INTO original_images (image_path) VALUES (?)", (image_path,))
        conn.commit()
        conn.close()
        st.success("原始图像已存储。")


# 存储中间结果
def store_intermediate_result(result_type, result_data):
    conn = sqlite3.connect('data_storage.db')
    c = conn.cursor()
    c.execute("INSERT INTO intermediate_results (result_type, result_data) VALUES (?,?)", (result_type, result_data))
    conn.commit()
    conn.close()
    st.success(f"{result_type} 中间结果已存储。")


# 存储最终结果（综合存储）
def store_final_result(result_type, result_data):
    conn = sqlite3.connect('data_storage.db')
    c = conn.cursor()
    c.execute("INSERT INTO final_results (result_type, result_data) VALUES (?,?)", (result_type, result_data))
    conn.commit()
    conn.close()
    st.success("最终结果已存储。")


# 查看数据库中的数据
def view_database_data():
    conn = sqlite3.connect('data_storage.db')
    c = conn.cursor()
    st.subheader("查看原始图像表数据")
    c.execute("SELECT * FROM original_images")
    original_images = c.fetchall()
    st.table(original_images)  

    st.subheader("查看中间结果表数据")
    c.execute("SELECT * FROM intermediate_results")
    intermediate_results = c.fetchall()
    st.table(intermediate_results)

    st.subheader("查看最终结果表数据")
    c.execute("SELECT * FROM final_results")
    final_results = c.fetchall()
    st.table(final_results)

    conn.close()


# 显示图像
def display_image(image_path):
    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        st.image(image, caption="Uploaded Image", use_column_width=True)


# 筛选并查看数据
def filter_and_view_data():
    conn = sqlite3.connect('data_storage.db')
    c = conn.cursor()
    st.subheader("筛选并查看数据")
    table_choice = st.selectbox("选择要筛选的表", ["original_images", "intermediate_results", "final_results"])
    if table_choice == "original_images":
        st.write("按时间范围筛选原始图像数据")
        start_date = st.date_input("开始日期")
        end_date = st.date_input("结束日期")
        start_timestamp = datetime.combine(start_date, datetime.min.time()).timestamp()
        end_timestamp = datetime.combine(end_date, datetime.max.time()).timestamp()
        c.execute("SELECT * FROM original_images WHERE upload_time BETWEEN? AND?", (start_timestamp, end_timestamp))
    elif table_choice == "intermediate_results":
        st.write("按结果类型筛选中间结果数据")
        result_type = st.text_input("输入结果类型")
        c.execute("SELECT * FROM intermediate_results WHERE result_type =?", (result_type,))
    elif table_choice == "final_results":
        st.write("按结果类型筛选最终结果数据")
        result_type = st.text_input("输入结果类型")
        c.execute("SELECT * FROM final_results WHERE result_type =?", (result_type,))
    rows = c.fetchall()
    st.table(rows)
    conn.close()

# 清除数据库数据
def clear_database():
    conn = sqlite3.connect('data_storage.db')
    c = conn.cursor()
    # 删除表中的数据
    c.execute("DELETE FROM defect_detection")
    c.execute("DELETE FROM defect_classification")
    c.execute("DELETE FROM fault_diagnosis")
    c.execute("DELETE FROM original_images")
    c.execute("DELETE FROM intermediate_results")
    c.execute("DELETE FROM final_results")
    conn.commit()
    conn.close()
    st.success("数据库已清除")


def main():
    st.title("数据存储模块演示")
    init_db()

    # 存储原始图像
    st.subheader("存储原始图像")
    uploaded_image = st.file_uploader("上传原始图像", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        store_original_image(image)
        # 显示上传的图像
        display_image(os.path.join("images", f"original_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"))

    # 存储中间结果
    st.subheader("存储中间结果")
    st.info("可能存在缺陷的区域坐标、可疑像素点的特征值等、分割后的缺陷区域图像数据、分割算法使用的阈值等、图像特征提取过程中的特征向量等")
    result_type = st.text_input("输入中间结果类型")
    result_data = st.text_area("输入中间结果数据", height=100)
    if st.button("存储中间结果"):
        store_intermediate_result(result_type, result_data)

    # 存储最终结果（综合存储）
    st.subheader("存储最终结果")
    st.info("缺陷分类结果、图像识别最终结果、故障诊断最终结果...")
    # st.markdown('<p style="font-size: 12px;">（缺陷分类结果、图像识别最终结果、故障诊断最终结果...）</p>', unsafe_allow_html=True)
    result_type = st.text_input("输入最终结果类型", value="")
    result_data = st.text_area("输入最终结果数据", height=100)
    if st.button("存储最终结果"):
        store_final_result(result_type, result_data)

    # 将查看和筛选数据的按钮放入侧边栏
    
    selected_option = st.sidebar.selectbox("数据库操作",("请选择","查看数据库数据","筛选并查看数据","清除数据库"))
    # if st.sidebar.button("查看数据库数据"):
    #     view_database_data()
    # if st.sidebar.button("筛选并查看数据"):
    #     filter_and_view_data()
    if selected_option == "查看数据库数据":
        view_database_data()
    elif selected_option == "筛选并查看数据":
        filter_and_view_data()
    elif selected_option == "清除数据库":
        clear_database()


if __name__ == "__main__":
    main()