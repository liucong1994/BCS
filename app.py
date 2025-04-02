# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import xgboost

# 配置页面
st.set_page_config(
    page_title="BCS出血风险预测工具",
    page_icon=":hospital:",
    layout="wide"
)

@st.cache_resource
def load_assets():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(base_path, "assets")
        
        # 检查资源目录
        if not os.path.exists(assets_dir):
            raise FileNotFoundError(f"资源目录缺失: {assets_dir}")
            
        model_path = os.path.join(assets_dir, "bcs_hemorrhage_xgb_model.pkl")
        feature_names_path = os.path.join(assets_dir, "feature_names2.pkl")

        # 加载模型
        model = joblib.load(model_path)
        
        # 验证模型类型
        if not isinstance(model, xgboost.XGBClassifier):
            raise TypeError("模型类型错误：必须为XGBoost分类器")
        if model.n_classes_ != 2:
            raise ValueError("当前仅支持二分类模型")
            
        # 初始化SHAP解释器
        explainer = shap.TreeExplainer(
            model=model,
            model_output="probability",
            feature_perturbation="tree_path_dependent"
        )
        
        # 加载特征名称
        feature_names = joblib.load(feature_names_path)
        
        return model, explainer, feature_names
        
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        sys.exit(1)

model, explainer, feature_names = load_assets()

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

    try:
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

        # SHAP解释部分
        st.subheader("风险解释")
        with st.spinner("生成特征贡献度分析..."):
            # 计算SHAP值
            shap_values = explainer.shap_values(input_df)
            
            # 调试输出（部署后可查看日志）
            st.session_state['debug'] = {
                'expected_value': explainer.expected_value,
                'shap_values_shape': np.array(shap_values).shape,
                'model_type': str(type(model))
            }
            
            # 可视化配置
            plt.figure(figsize=(10, 4))
            
            # 二分类SHAP可视化适配
            if isinstance(explainer.expected_value, np.ndarray):
                base_value = explainer.expected_value[1]  # 获取正类的基准值
                sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            else:
                base_value = explainer.expected_value
                sv = shap_values[0]

            shap.force_plot(
                base_value,
                sv,
                input_df.iloc[0],
                matplotlib=True,
                show=False,
                text_rotation=15,
                plot_cmap=["#FF4B4B", "#2E86C1"]  # 自定义颜色
            )
            
            plt.title("特征贡献度分析", fontsize=14)
            plt.tight_layout()
            plt.savefig("shap_plot.png", dpi=120, bbox_inches='tight')
            plt.close()
            
            st.image("shap_plot.png")

    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")

# 页脚
st.markdown("---")
st.caption("© 预测工具仅限学术研究使用，不作为诊疗唯一依据。")
