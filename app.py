import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 页面配置 (暗黑模式宽屏)
st.set_page_config(page_title="AI 音乐风投评估终端", layout="wide", initial_sidebar_state="expanded")


# 强制加载我们刚才打包的文件
@st.cache_resource
def load_models():
    model = joblib.load('xgb_model.pkl')
    genre_data = joblib.load('genre_dict.pkl')
    return model, genre_data


model, genre_data = load_models()
genre_dict = genre_data['dict']
global_mean = genre_data['global_mean']

# 辅助函数：厂牌与流派判定
major_labels = ['columbia', 'warner', 'atlantic', 'epic', 'emi', 'elektra', 'reprise', 'island', 'virgin', 'capitol']
famous_indie = ['prestige', 'blue note', 'ecm', 'impulse', 'tamla', '4ad', 'sub pop', 'relapse', 'creation', 'merge',
                'kranky', 'warp']


def get_tier(label):
    l = str(label).lower()
    if any(m in l for m in major_labels): return 2
    if any(i in l for i in famous_indie): return 1
    return 0


# --- UI 界面开始 ---
st.title("🎵 AI 音乐风投量化评估终端")
st.markdown("**基于 XGBoost & K-Fold 目标编码 | 严禁数据泄露 | 锁定 0.50 黄金决策阈值**")
st.divider()

col1, col2 = st.columns([1, 1.5])

# === 左侧：输入控制台 ===
with col1:
    st.subheader("💿 商业元数据萃取")
    album_name = st.text_input("项目/专辑名称", "Brat")

    c1, c2 = st.columns(2)
    with c1: primary_genre = st.text_input("原生第一子流派", "dance-pop")
    with c2: label_input = st.text_input("发行厂牌", "atlantic")

    secondary_genres = st.text_input("融合流派 (逗号分隔，测算复杂度)", "electropop, hyperpop, club")

    st.subheader("🎙️ 物理声学矩阵")
    danceability = st.slider("可舞度 (Danceability)", 0.0, 1.0, 0.75, 0.01)
    energy = st.slider("能量感 (Energy)", 0.0, 1.0, 0.85, 0.01)
    speechiness = st.slider("言语性 (Speechiness)", 0.0, 1.0, 0.15, 0.01)
    liveness = st.slider("现场感 (Liveness)", 0.0, 1.0, 0.20, 0.01)
    valence = st.slider("愉悦度 (Valence)", 0.0, 1.0, 0.60, 0.01)
    acousticness = st.slider("原声度 (Acousticness)", 0.0, 1.0, 0.10, 0.01)
    instrumentalness = st.slider("器乐度 (Instrumentalness)", 0.0, 1.0, 0.05, 0.01)
    tempo = st.slider("速度 (Tempo/BPM)", 50.0, 200.0, 120.0, 1.0)
    duration_ms = st.number_input("时长 (毫秒)", value=210000)

# === 数据预处理与预测计算 ===
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

# 模型预测
# 修改后（加一个 float 强制转换）：
prob = float(model.predict_proba(input_df)[0][1])
is_hit = prob >= 0.50

# === 右侧：输出看板 ===
with col2:
    st.subheader("📊 决策雷达与归因分析")

    # 核心得分卡片
    if is_hit:
        st.success(f"### 投资建议：越过 0.50 黄金线，建议重点关注！")
    else:
        st.error(f"### 投资建议：低于大盘期望，建议规避风险！")

    st.metric(label="AI 预测双高爆款概率", value=f"{prob * 100:.1f}%",
              delta=f"{(prob - 0.262) * 100:.1f}% 跑赢大盘" if prob > 0.262 else f"{(prob - 0.262) * 100:.1f}% 落后大盘")
    st.progress(prob)

    st.markdown("---")
    st.markdown("#### 🔍 特征归因诊断 (输入层)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**第一流派历史胜率 (权重榜首):**\n\n【{clean_genre}】 历史平滑胜率 {target_enc * 100:.1f}%")
        st.warning(f"**先锋性惩罚机制:**\n\n包含 {complexity} 个标签融合")

    with col_b:
        tier_str = "Tier 1 (精英独立)" if tier == 1 else "Tier 2 (三大巨头)" if tier == 2 else "Tier 0 (普通作坊)"
        st.success(f"**厂牌资本评级 (弱修正):**\n\n判定为 {tier_str}")
        st.info(f"**声学甜点区探测:**\n\n多维非线性矩阵计算中...")
