import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==========================================
# 1. 页面与全局配置
# ==========================================
st.set_page_config(page_title="AI 音乐风投量化评估终端", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 2. 加载模型与字典数据
# ==========================================
@st.cache_resource
def load_models():
    # 确保本地有这俩文件，由 export_model.py 生成
    model = joblib.load('xgb_model.pkl')
    genre_data = joblib.load('genre_dict.pkl')
    return model, genre_data

model, genre_data = load_models()
genre_dict = genre_data['dict']
global_mean = genre_data['global_mean']

# ==========================================
# 3. 辅助函数定义
# ==========================================
major_labels = ['columbia', 'warner', 'atlantic', 'epic', 'emi', 'elektra', 'reprise', 'island', 'virgin', 'capitol']
famous_indie = ['prestige', 'blue note', 'ecm', 'impulse', 'tamla', '4ad', 'sub pop', 'relapse', 'creation', 'merge', 'kranky', 'warp']

def get_tier(label):
    l = str(label).lower()
    if any(m in l for m in major_labels): return 2
    if any(i in l for i in famous_indie): return 1
    return 0

def generate_acoustic_report(da, en, sp, lv, ac, ins):
    """根据输入的声学特征，动态生成严谨的声学评估短评"""
    report = []
    if da > 0.7: report.append("显著的律动性 (高可舞度)")
    if en > 0.75: report.append("强烈的声场能量释放")
    if sp > 0.3: report.append("较高的人声/言语占比")
    if lv > 0.5: report.append("现场混音特征明显")
    if ac > 0.6: report.append("原声乐器主导")
    if ins > 0.6: report.append("纯器乐属性，缺乏人声流行度")
    
    if not report:
        return "各项声学指标均衡，无极端物理属性偏向。"
    return "、".join(report) + "。"

# ==========================================
# 4. 前端 UI 构建
# ==========================================
st.title("🎵 AI 音乐风投量化评估终端")
st.markdown("基于 **XGBoost & K-Fold 折外目标编码** 构建 | 严防数据泄露 | **锁定 0.50 决策阈值**")
st.divider()

col1, col2 = st.columns([1, 1.3], gap="large")

# ---------------- 左侧：数据输入控制台 ----------------
with col1:
    st.subheader("💿 商业元数据萃取")
    album_name = st.text_input("项目/专辑名称", "Brat")
    
    c1, c2 = st.columns(2)
    with c1: 
        primary_genre = st.text_input("原生第一子流派", "dance-pop")
    with c2: 
        label_input = st.text_input("发行厂牌", "atlantic")
    
    secondary_genres = st.text_input("融合流派 (逗号分隔，用于计算复杂度)", "electropop, hyperpop, club")
    
    st.subheader("🎙️ 物理声学矩阵输入")
    danceability = st.slider("可舞度 (Danceability)", 0.0, 1.0, 0.75, 0.01)
    energy = st.slider("能量感 (Energy)", 0.0, 1.0, 0.85, 0.01)
    speechiness = st.slider("言语性 (Speechiness)", 0.0, 1.0, 0.15, 0.01)
    liveness = st.slider("现场感 (Liveness)", 0.0, 1.0, 0.20, 0.01)
    valence = st.slider("愉悦度 (Valence)", 0.0, 1.0, 0.60, 0.01)
    acousticness = st.slider("原声度 (Acousticness)", 0.0, 1.0, 0.10, 0.01)
    instrumentalness = st.slider("器乐度 (Instrumentalness)", 0.0, 1.0, 0.05, 0.01)
    tempo = st.slider("速度 (Tempo/BPM)", 50.0, 200.0, 120.0, 1.0)
    duration_ms = st.number_input("时长 (毫秒)", value=210000, step=1000)

# ---------------- 数据清洗与预测计算 ----------------
clean_genre = str(primary_genre).split(',')[0].strip().lower()
target_enc = genre_dict.get(clean_genre, global_mean)
tier = get_tier(label_input)
complexity = len([g for g in str(secondary_genres).split(',') if g.strip() != ''])

input_df = pd.DataFrame([{
    'danceability': danceability, 'energy': energy, 'speechiness': speechiness,
    'acousticness': acousticness, 'instrumentalness': instrumentalness, 'liveness': liveness,
    'valence': valence, 'tempo': tempo, 'duration_ms': duration_ms,
    'genre_complexity': complexity, 'label_tier': tier, 'genre_target_enc': target_enc
}])

prob = float(model.predict_proba(input_df)[0][1])
is_hit = prob >= 0.50

# ---------------- 右侧：输出看板与归因诊断 ----------------
with col2:
    st.subheader("📊 预测引擎仪表盘")
    
    # 绘制高级 Plotly 仪表盘
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prob * 100,
        number = {'suffix': "%", 'font': {'size': 60}},
        delta = {'reference': global_mean * 100, 'position': "top", 'increasing': {'color': "#2ecc71"}, 'decreasing': {'color': "#e74c3c"}},
        title = {'text': "双高爆款预测概率<br><span style='font-size:0.8em;color:gray'>对比大盘基础胜率 (26.2%)</span>", 'font': {"size": 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#6366f1" if is_hit else "#94a3b8"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 50], 'color': "rgba(231, 76, 60, 0.1)"},
                {'range': [50, 100], 'color': "rgba(46, 204, 113, 0.1)"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig, use_container_width=True)

    # 投资建议横幅
    if is_hit:
        st.success(f"**最终量化决策：** 越过 0.50 投资阈值线，期望收益高于大盘均值，建议纳入风投重点观察池。")
    else:
        st.error(f"**最终量化决策：** 未越过 0.50 投资阈值线，爆发期望极低，依据风投风控原则建议规避。")
        
    st.markdown("### 🔍 特征归因诊断 (输入层)")
    
    # 使用 container(border=True) 替代难看的 info 框，打造卡片感
    c_left, c_right = st.columns(2)
    
    with c_left:
        with st.container(border=True):
            st.markdown(f"**核心基准：流派 OOF 目标编码**")
            st.markdown(f"当前流派：`{clean_genre}`")
            st.markdown(f"历史折外平滑胜率：**{target_enc*100:.1f}%**")
            st.caption("注：该特征在 XGBoost 模型中贡献度占比最高。")
            
        with st.container(border=True):
            st.markdown(f"**结构评估：流派复杂度**")
            st.markdown(f"融合流派数量：**{complexity} 个**")
            st.caption("模型统计显示：适度融合有利于定位，过度融合易导致受众失焦。")

    with c_right:
        with st.container(border=True):
            tier_str = "Tier 1 (知名独立厂牌)" if tier == 1 else "Tier 2 (三大唱片巨头)" if tier == 2 else "Tier 0 (常规独立厂牌)"
            st.markdown(f"**辅助修正：厂牌资源等级**")
            st.markdown(f"资源评定：**{tier_str}**")
            st.caption("提供宣发与品控的基础基线修正。")

        with st.container(border=True):
            st.markdown(f"**动态测算：声学特征矩阵评估**")
            acoustic_report = generate_acoustic_report(danceability, energy, speechiness, liveness, acousticness, instrumentalness)
            st.markdown(f"表现特征：**{acoustic_report}**")
            st.caption("结合各项物理参数输入的综合听感研判。")
