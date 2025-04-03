# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os

# 配置Matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置页面
st.set_page_config(
    page_title="BCS出血风险预测工具",
    page_icon=":hospital:",
    layout="wide"
)

@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "assets", "bcs_hemorrhage_xgb_model.pkl")
    feature_names_path = os.path.join(base_path, "assets", "feature_names2.pkl")

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_names

model, explainer, feature_names = load_assets()

def st_shap(plot, height=None):
    """在Streamlit中显示SHAP plot的HTML组件"""
    shap_html = f"""
    <html>
    <head>
    {shap.getjs()}
    </head>
    <body>
    {shap.initjs()}
    {plot.html()}
    </body>
    </html>
    """
    components.html(shap_html, height=height)

st.title("布加综合征上消化道出血风险预测")
st.markdown("""**使用说明**:  
输入患者的基线指标，点击"预测"按钮获取6个月内出血风险及个体化建议。  
模型基于随机森林算法构建，并通过SHAP值解释预测依据。""")

with st.sidebar:
    st.header("患者指标输入")
    input_values = []
    for feature in feature_names:
        if "NLR" in feature:
            val = st.number_input(f"{feature}", min_value=0.1, value=3.5, step=0.1)
        elif "血小板" in feature:
            val = st.number_input(f"{feature} (×10⁹/L/cm)", min_value=0.1, value=18.0, step=0.1)
        elif "门静脉" in feature:
            val = st.number_input(f"{feature} (mm)", min_value=5.0, value=14.0, step=0.1)
        elif "IV型胶原" in feature:
            val = st.number_input(f"{feature} (ng/mL)", min_value=0.0, value=200.0, step=1.0)
        input_values.append(val)
    predict_button = st.button("开始预测")

if predict_button:
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # 预测概率
    prob = model.predict_proba(input_df)[0][1]
    risk_percent = round(prob * 100, 2)

    # 风险分级
    if risk_percent >= 30:
        risk_level = "高危"
        advice = "立即住院监测，优先安排内镜检查，考虑预防性TIPS"
        color = "#FF4B4B"
    elif risk_percent >= 10:
        risk_level = "中危"
        advice = "每2周门诊随访，启动非选择性β受体阻滞剂治疗"
        color = "#FFA500"
    else:
        risk_level = "低危"
        advice = "每3个月常规随访，维持抗凝治疗"
        color = "#2E86C1"

    # 显示预测结果
    st.subheader("预测结果")
    st.markdown(f"""
    <div style="border-left: 5px solid {color}; padding: 10px;">
        <h4 style="color: {color};">{risk_level}风险</h4>
        <p>6个月内出血概率: <b>{risk_percent:.2f}%</b></p>
        <p>临床建议: {advice}</p>
    </div>
    """, unsafe_allow_html=True)

    # SHAP解释
    st.subheader("预测解释")
    with st.spinner("生成SHAP解释..."):
        # 计算SHAP值
        shap_values = explainer.shap_values(input_df)
        
        # 生成HTML格式的force plot
        force_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0],
            features=input_df.iloc[0, :],
            feature_names=feature_names,
            show=False
        )
        st_shap(force_plot, height=300)

    # 指标说明
    st.markdown("---")
    st.subheader("指标临床意义")
    st.markdown("""
    - **NLR**: 反映全身炎症状态  
    - **血小板/脾脏比值**: 门静脉高压严重程度指标  
    - **门静脉宽度**: 门静脉高压严重程度指标  
    - **IV型胶原**: 肝纤维化标志物  
    """)

# 页脚
st.markdown("---")
st.caption("© 2024 布加综合征研究组。预测工具仅限临床医生使用，不作为诊疗唯一依据。")
