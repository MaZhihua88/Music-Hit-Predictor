import streamlit as st
import joblib
import pandas as pd

# ==========================================
# 1. 页面基础设置
# ==========================================
st.set_page_config(page_title="AI 音乐风投引擎", page_icon="🎵", layout="wide")
st.title("🎵 AI 音乐爆款潜力预测引擎 (V2.0 纯净版)")
st.markdown(
    "输入您的专辑原始数据，系统将在后台自动进行**文本解析与特征转换**，并基于工业级模型 (Threshold=0.50) 生成风投诊断报告。")
st.markdown("---")


# ==========================================
# 2. 加载最新版模型
# ==========================================
@st.cache_resource
def load_model():
    # ⚠️ 请确保路径是你保存的那个没有任何随机漂移的 pkl 文件
    return return joblib.load('music_hit_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ 模型加载失败，请检查路径: {e}")
    st.stop()

# ==========================================
# 3. 用户界面：输入“人话” (原始数据)
# ==========================================
col_input, col_spacer, col_output = st.columns([4, 1, 5])

with col_input:
    st.subheader("📝 1. 专辑原始信息录入")

    st.markdown("**🏢 商业与文化信息**")
    raw_label = st.text_input("厂牌名称 (例如: 4AD, Columbia, 或自己工作室的名字)", value="独立作坊")

    # 预设主流派字典 (使用我们训练集大盘均值兜底 0.26)
    genre_dict = {"摇滚 (Rock)": 0.28, "流行 (Pop)": 0.22, "爵士 (Jazz)": 0.45, "民谣 (Folk)": 0.35,
                  "电子 (Electronic)": 0.25, "其他 (Other)": 0.26}
    raw_primary_genre = st.selectbox("主流派 (Primary Genre)", list(genre_dict.keys()))

    raw_secondary_genre = st.text_input("融合流派标签 (请用逗号分隔，例如: Indie Rock, Ambient)", value="")

    st.markdown("**🎛️ 声学物理参数**")
    energy = st.slider("⚡ 能量感 (Energy)", 0.0, 1.0, 0.60)
    danceability = st.slider("💃 律动感 (Danceability)", 0.0, 1.0, 0.50)
    tempo = st.slider("🥁 速度 (Tempo BPM)", 60.0, 200.0, 120.0)
    acousticness = st.slider("🎻 原声度 (Acousticness)", 0.0, 1.0, 0.20)
    instrumentalness = st.slider("🎸 器乐化程度 (Instrumentalness)", 0.0, 1.0, 0.0)
    liveness = st.slider("🎤 现场感 (Liveness)", 0.0, 1.0, 0.10)
    valence = st.slider("😊 愉悦度 (Valence)", 0.0, 1.0, 0.50)
    speechiness = st.slider("🗣️ 人声密集度 (Speechiness)", 0.0, 1.0, 0.05)
    duration_ms = st.number_input("⏱️ 时长 (毫秒)", value=210000, step=10000)

# ==========================================
# 4. 🤖 黑盒逻辑：后台自动特征工程
# ==========================================
# A. 厂牌降维映射
major_labels = ['columbia', 'warner', 'atlantic', 'epic', 'emi', 'elektra', 'reprise', 'island', 'virgin', 'capitol']
famous_indie = ['prestige', 'blue note', 'ecm', 'impulse', 'tamla', '4ad', 'sub pop', 'relapse', 'creation', 'merge',
                'kranky', 'warp']
raw_label_lower = raw_label.lower()
if any(m in raw_label_lower for m in major_labels):
    label_tier = 2
elif any(i in raw_label_lower for i in famous_indie):
    label_tier = 1
else:
    label_tier = 0

# B. 流派复杂度自动计算 (数逗号)
genre_complexity = 0 if raw_secondary_genre.strip() == "" else len(raw_secondary_genre.split(','))

# C. 流派热度自动映射
genre_target_enc = genre_dict[raw_primary_genre]

# 严格按照训练集的顺序组装数据！(无 cluster_id)
input_data = {
    'danceability': [danceability], 'energy': [energy], 'speechiness': [speechiness],
    'acousticness': [acousticness], 'instrumentalness': [instrumentalness],
    'liveness': [liveness], 'valence': [valence], 'tempo': [tempo],
    'duration_ms': [duration_ms], 'genre_complexity': [genre_complexity],
    'label_tier': [label_tier], 'genre_target_enc': [genre_target_enc]
}
input_df = pd.DataFrame(input_data)

# ==========================================
# 5. 预测与报告 (阈值强锁 0.50)
# ==========================================
prob = model.predict_proba(input_df)[0][1]
FIXED_THRESHOLD = 0.50

with col_output:
    st.subheader("📊 2. AI 实时诊断报告")
    st.metric(label="双高爆款概率 (投资安全指数)", value=f"{prob * 100:.1f}%",
              delta=f"{prob * 100 - FIXED_THRESHOLD * 100:.1f}% (距及格线)")

    if prob >= FIXED_THRESHOLD:
        st.success(f"🔥 **极具爆款潜力！** AI 判断这首歌跨越了 {FIXED_THRESHOLD * 100:.0f}% 的风投安全线。")
    else:
        st.warning(f"🤔 **未达及格线。** 建议参考下方修改意见。")

    st.markdown("---")
    st.subheader("💡 隐藏特征透视 (开发者专属)")
    st.caption(f"系统已自动将您的输入转化为 AI 专属特征：")
    st.code(
        f"• 厂牌级别 (Label Tier): {label_tier}\n• 流派复杂度 (Complexity): {genre_complexity}\n• 流派基础热度 (Target Enc): {genre_target_enc:.3f}")

    st.subheader("🛠️ 智能混音与宣发建议")
    if prob >= FIXED_THRESHOLD:
        st.info("✨ 数据结构已达标！请保持当前制作方向。")
    else:
        if label_tier == 0:
            st.error("🤝 **商业背书不足**：独立作坊极难出头，建议将 Demo 投递给 Tier 1 或 Tier 2 的成熟厂牌。")
        if genre_complexity < 2:
            st.warning("🎼 **缺乏先锋性**：您的风格标签过于单一，尝试在副歌加入其他风格的器乐元素进行融合。")
        if energy < 0.5 and tempo > 120:
            st.error("⚠️ **听感矛盾**：BPM 极高但能量感微弱，这会让听众感到乏力，建议推高混音的声压级。")
