# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import shap
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import os

# 初始化SHAP JS库 (必须最先执行)
shap.initjs()

# 配置页面
st.set_page_config(
    page_title="BCS出血风险预测系统",
    page_icon="⚕️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """加载预训练模型和特征"""
    base_path = os.path.dirname(__file__)
    return (
        joblib.load(os.path.join(base_path, "assets", "bcs_hemorrhage_xgb_model.pkl")),
        joblib.load(os.path.join(base_path, "assets", "feature_names2.pkl"))
    )

model, feature_names = load_model()

# 界面布局
st.title("🩺 布加综合征出血风险预测")
st.markdown("""
<style>
.big-font { font-size:18px !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("患者指标录入")
    inputs = []
    for idx, name in enumerate(feature_names):
        if "NLR" in name:
            inputs.append(st.number_input(f"{name}", 3.5, 0.1, 50.0, 0.1))
        elif "血小板" in name:
            inputs.append(st.number_input(f"{name} (×10⁹/L/cm)", 18.0, 0.1, 100.0, 0.1))
        elif "门静脉" in name:
            inputs.append(st.number_input(f"{name} (mm)", 14.0, 5.0, 30.0, 0.1))
        elif "IV型胶原" in name:
            inputs.append(st.number_input(f"{name} (ng/mL)", 200.0, 0.0, 1000.0, 1.0))
    
    if st.button("开始风险评估", use_container_width=True):
        st.session_state.predict = True

if st.session_state.get('predict', False):
    # 数据预处理
    input_data = pd.DataFrame([inputs], columns=feature_names)
    
    # 风险预测
    proba = model.predict_proba(input_data)[0][1]
    risk_level = (
        "高危" if (proba >= 0.3) else
        "中危" if (proba >= 0.1) else "低危"
    )
    
    # 结果展示
    st.success(f"### 风险评估结果：{risk_level}")
    st.metric(label="6个月内出血概率", value=f"{proba*100:.1f}%")
    
    # SHAP可视化 (核心修改部分)
    st.markdown("---")
    with st.spinner("生成个性化解释..."):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            # 生成交互式可视化
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                {shap.getjs()}
                <meta charset="utf-8">
            </head>
            <body style="margin:0;padding:10px;">
                {shap.force_plot(
                    base_value=explainer.expected_value,
                    shap_values=shap_values,
                    features=input_data.iloc[0,:],
                    feature_names=feature_names,  # 中文特征名
                    text_rotation=25,             # 调整标签角度
                    figsize=(14, 6),               # 调整显示尺寸
                    contribution_threshold=0.05,  # 过滤微小贡献值
                    plot_cmap=["#FF4040", "#00CD00"]  # 红绿配色
                ).html()}
            </body>
            </html>
            """
            
            # 渲染组件
            components.html(html_content, height=450, scrolling=False)
            
        except Exception as e:
            st.error(f"可视化生成失败：{str(e)}")
            st.error("建议：1.检查模型兼容性 2.确认输入格式正确")

    # 指标解读
    st.markdown("---")
    with st.expander("📖 临床指标解读"):
        st.markdown("""
        - **NLR（中性粒细胞/淋巴细胞比值）**  
          全身炎症反应标志物，>3.5提示炎症激活
        - **血小板/脾脏比值**  
          门静脉高压核心指标，<20提示脾功能亢进
        - **门静脉宽度**  
          >13mm提示显著门脉高压
        - **IV型胶原**  
          肝纤维化特异性标志物，>180ng/mL需警惕
        """)

# 页脚信息
st.markdown("---")
st.caption("""
⚠️ 注意事项：  
1. 本工具预测结果仅供参考，不作为临床决策唯一依据  
2. 模型基于XGBoost(v1.6)构建，SHAP版本0.44.0  
3. 临床数据更新至2023年12月  
""")
