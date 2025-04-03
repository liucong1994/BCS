# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import shap
import pandas as pd
import streamlit.components.v1 as components
import os

# 删除或注释掉以下行
# shap.initjs()  # 此方法仅适用于Jupyter环境

@st.cache_resource
def load_assets():
    """加载模型和特征名称"""
    base_path = os.path.dirname(__file__)
    return (
        joblib.load(os.path.join(base_path, "assets", "bcs_hemorrhage_xgb_model.pkl")),
        joblib.load(os.path.join(base_path, "assets", "feature_names2.pkl"))
    )

model, feature_names = load_assets()

# 界面布局
st.title("🩺 布加综合征出血风险预测系统")

with st.sidebar:
    st.header("患者指标录入")
    inputs = [
        st.number_input("NLR", 3.5, 0.1, 50.0, 0.1),
        st.number_input("血小板/脾脏比值 (×10⁹/L/cm)", 18.0, 0.1, 100.0, 0.1),
        st.number_input("门静脉宽度 (mm)", 14.0, 5.0, 30.0, 0.1),
        st.number_input("IV型胶原 (ng/mL)", 200.0, 0.0, 1000.0, 1.0)
    ]
    
if st.button("开始评估", type="primary"):
    # 预测逻辑
    input_df = pd.DataFrame([inputs], columns=feature_names)
    proba = model.predict_proba(input_df)[0][1]
    
    # SHAP可视化修复部分
    with st.spinner("生成解释..."):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            # 使用安全渲染模式
            components.html(
                shap.force_plot(
                    base_value=explainer.expected_value,
                    shap_values=shap_values,
                    features=input_df.iloc[0],
                    feature_names=feature_names,
                    matplotlib=False  # 关键修改：禁用matplotlib模式
                ).html(),
                height=500
            )
            
        except Exception as e:
            st.error(f"可视化生成失败: {str(e)}")

    # 显示结果
    st.success(f"预测风险值: {proba*100:.1f}%")
