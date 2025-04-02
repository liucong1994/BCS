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
from sklearn.base import ClassifierMixin

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
        
        # 严格检查资源路径
        if not os.path.exists(assets_dir):
            raise FileNotFoundError(f"关键目录缺失: {assets_dir}")
            
        model_path = os.path.join(assets_dir, "bcs_hemorrhage_xgb_model.pkl")
        feature_names_path = os.path.join(assets_dir, "feature_names2.pkl")

        # 验证模型文件存在
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型并验证类型
        model = joblib.load(model_path)
        if not isinstance(model, (xgboost.XGBClassifier, ClassifierMixin)):
            raise TypeError("无效模型类型，必须为分类器")
        if not hasattr(model, 'n_classes_') or model.n_classes_ != 2:
            raise ValueError("仅支持二分类模型")
            
        # 配置SHAP解释器（关键参数修正）
        explainer = shap.TreeExplainer(
            model=model,
            feature_perturbation="tree_path_dependent",
            model_output="raw"  # 必须使用原始输出
        )
        
        # 加载特征名称
        feature_names = joblib.load(feature_names_path)
        if len(feature_names) != model.n_features_in_:
            raise ValueError("特征数量不匹配")
            
        return model, explainer, feature_names
        
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        sys.exit(1)

model, explainer, feature_names = load_assets()

# 界面布局
st.title("布加综合征上消化道出血风险预测")
st.markdown("""**使用说明**: 输入指标后点击预测按钮，获取6个月内出血风险评估""")

# 侧边栏输入
with st.sidebar:
    st.header("患者指标录入")
    input_data = {}
    for feat in feature_names:
        if "NLR" in feat:
            val = st.number_input(f"{feat}", 0.1, 50.0, 3.5, 0.1)
        elif "血小板" in feat:
            val = st.number_input(f"{feat} (×10⁹/L/cm)", 0.1, 300.0, 18.0, 0.1)
        elif "门静脉" in feat:
            val = st.number_input(f"{feat} (mm)", 5.0, 50.0, 14.0, 0.1)
        elif "IV型胶原" in feat:
            val = st.number_input(f"{feat} (ng/mL)", 0.0, 1000.0, 200.0, 1.0)
        input_data[feat] = val
    predict_btn = st.button("执行预测")

def sigmoid(x):
    """将原始分数转换为概率"""
    return 1 / (1 + np.exp(-x))

if predict_btn:
    try:
        # 转换输入数据
        input_df = pd.DataFrame([input_data.values()], columns=feature_names)
        
        # 获取原始预测分数
        raw_score = model.predict_proba(input_df, output_margin=True)[0][1]
        
        # 转换为概率
        probability = sigmoid(raw_score)
        risk_percent = round(probability * 100, 2)
        
        # 风险分级逻辑
        if risk_percent >= 30:
            risk_level = ("高危", "#FF4B4B", "立即住院监测，优先内镜检查")
        elif risk_percent >= 10:
            risk_level = ("中危", "#FFA500", "每2周随访，β受体阻滞剂治疗")
        else:
            risk_level = ("低危", "#2E86C1", "每3个月常规随访")
            
        # 显示预测结果
        st.subheader("风险评估结果")
        st.markdown(f"""
        <div style="border-left: 5px solid {risk_level[1]}; padding: 10px;">
            <h3 style="color: {risk_level[1]};">{risk_level[0]}风险</h3>
            <p>6个月出血概率: <b>{risk_percent}%</b></p>
            <p>临床建议: {risk_level[2]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 生成SHAP解释
        with st.spinner("生成特征贡献分析..."):
            plt.figure(figsize=(10, 4))
            
            # 计算SHAP值
            shap_values = explainer.shap_values(input_df)
            
            # 可视化参数处理
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            sv = shap_values[0][1] if isinstance(shap_values, list) else shap_values[0]
            
            # 绘制SHAP力图
            shap.force_plot(
                base_value,
                sv,
                input_df.iloc[0],
                matplotlib=True,
                show=False,
                text_rotation=15,
                plot_cmap=["#FF4B4B", "#2E86C1"]
            )
            
            plt.title("各特征对预测结果的贡献度", pad=20)
            plt.tight_layout()
            plt.savefig("shap_plot.png", dpi=120, bbox_inches='tight')
            plt.close()
            
            st.image("shap_plot.png")
            
    except Exception as e:
        st.error(f"预测过程中出现异常: {str(e)}")

# 页脚信息
st.markdown("---")
st.caption("注：本工具预测结果仅供参考，临床决策需结合其他检查")
