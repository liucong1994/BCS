# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import shap
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import os

# 配置页面
st.set_page_config(
    page_title="BCS出血风险预测工具",
    page_icon=":hospital:",
    layout="wide"
)

@st.cache_resource
def load_assets():
    """加载模型和特征名称"""
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "assets", "bcs_hemorrhage_xgb_model.pkl")
    feature_names_path = os.path.join(base_path, "assets", "feature_names2.pkl")

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)  # 确保此处加载的是中文特征名
    return model, feature_names

model, feature_names = load_assets()

st.title("布加综合征上消化道出血风险预测")
st.markdown("""**使用说明**:  
输入患者的基线指标，点击"预测"按钮获取6个月内出血风险及个体化建议。""")

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
    
    # 预测逻辑
    prob = model.predict_proba(input_df)[0][1]
    risk_percent = round(prob * 100, 2)

    # 风险分级逻辑
    if risk_percent >= 30:
        risk_level = "高危"
        advice = "立即住院监测，优先安排内镜检查"
    elif risk_percent >= 10:
        risk_level = "中危"
        advice = "每2周门诊随访"
    else:
        risk_level = "低危"
        advice = "每3个月常规随访"

    # 显示预测结果
    st.success(f"风险评估结果：{risk_level}（{risk_percent}% 概率）")
    st.info(f"临床建议：{advice}")

    # ==================== 关键修改部分 ====================
    st.subheader("预测解释")
    with st.spinner("生成SHAP解释..."):
        try:
            # 初始化SHAP解释器
            explainer = shap.TreeExplainer(model)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(input_df)
            
            # 生成JavaScript可视化
            shap_html = f"""
            <html>
                <head>
                    {shap.getjs()}
                </head>
                <body style="margin:0;padding:0;">
                    {shap.force_plot(
                        base_value=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                        shap_values=shap_values[1] if isinstance(shap_values, list) else shap_values[0],
                        features=input_df.iloc[0].values,
                        feature_names=feature_names,  # 直接使用中文特征名
                        text_rotation=30,  # 调整文字角度
                        figsize=(12, 6)     # 调整显示尺寸
                    ).html()}
                </body>
            </html>
            """
            
            # 渲染组件
            components.html(shap_html, height=500, scrolling=True)
            
        except Exception as e:
            st.error(f"解释生成失败: {str(e)}")
    # ==================== 修改结束 ====================

    # 指标说明
    st.markdown("---")
    st.subheader("指标说明")
    st.markdown("""
    - **NLR**: 中性粒细胞/淋巴细胞比值  
    - **血小板/脾脏比值**: 门脉高压指标  
    - **门静脉宽度**: 血管压力指标  
    - **IV型胶原**: 肝纤维化标志物  
    """)

# 页脚
st.markdown("---")
st.caption("© 2024 临床预测模型研究组")
