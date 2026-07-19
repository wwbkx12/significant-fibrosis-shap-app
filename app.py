"""
基于 Streamlit 的显著肝纤维化预测应用。

功能：
1. 接收 9 个数值型特征；
2. 调用已经训练好的 SVM 模型预测 Outcome=1 的概率；
3. 使用 SHAP KernelExplainer 解释当前这个病例的预测结果；
4. 在网页中显示概率和 SHAP force plot。

运行前请把 svm_model.pkl 和 shap_background.csv 放在本文件同一目录。
"""

# ==================== 第 1 步：导入需要的 Python 包 ====================
# pathlib：用可靠的方式拼接文件路径，避免 Windows 和 Linux 路径分隔符不同。
# pickle：读取本地保存的 svm_model.pkl 模型。
# matplotlib：承载 SHAP 生成的静态解释图。
# numpy / pandas：整理数值和表格数据。
# shap：解释单个预测结果。
# streamlit：创建网页界面。
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ==================== 第 2 步：设置网页和文件路径 ====================
st.set_page_config(
    page_title="Prediction Model with SHAP Visualization",
    page_icon="🩺",
    layout="centered",
)

# BASE_DIR 表示 app.py 所在目录。
# Streamlit Community Cloud 运行程序时的工作目录可能发生变化，因此不要只写相对路径。
BASE_DIR = Path(__file__).resolve().parent

# 【需要修改 1】如果文件名不同，只修改下面两个文件名，不要修改其他读取代码。
MODEL_PATH = BASE_DIR / "svm_model.pkl"
BACKGROUND_PATH = BASE_DIR / "shap_background.csv"

# 特征名称及顺序必须与训练模型时 X_train 的列名称和列顺序完全一致。
# 即使网页输入顺序看起来无关紧要，送入模型时的顺序也绝对不能改变。
FEATURE_NAMES = [
    "Age",
    "PLT",
    "FBG",
    "ALT",
    "AST",
    "GGT",
    "FINS",
    "UAR",
    "TyG.WHtR",
]

# 【需要修改 2】这里控制每个数字输入框的最小值、最大值、默认值和步长。
# 当前默认值取自本项目 traindata.csv 的中位数；范围略宽于训练集实际范围。
# 这些值用于限制明显错误的输入，不代表医学参考区间，也不用于诊断。
# 如果以后更换训练数据、变量单位或模型，应按新数据重新修改。
INPUT_CONFIG = {
    "Age": {"min": 18.0, "max": 100.0, "default": 51.5, "step": 1.0},
    "PLT": {"min": 10.0, "max": 500.0, "default": 229.0, "step": 1.0},
    "FBG": {"min": 1.0, "max": 25.0, "default": 7.74, "step": 0.1},
    "ALT": {"min": 0.0, "max": 1000.0, "default": 18.52, "step": 1.0},
    "AST": {"min": 0.0, "max": 1000.0, "default": 18.97, "step": 1.0},
    "GGT": {"min": 0.0, "max": 1000.0, "default": 27.55, "step": 1.0},
    "FINS": {"min": 0.0, "max": 30.0, "default": 7.75, "step": 0.1},
    "UAR": {"min": 0.0, "max": 35.0, "default": 7.54, "step": 0.1},
    "TyG.WHtR": {"min": 0.0, "max": 12.0, "default": 4.59, "step": 0.01},
}

# nsamples 是解释一个病例时 Kernel SHAP 使用的采样次数。
# 如果网页运行较慢，可改成 100；如果更重视解释稳定性，可改成 300～500。
SHAP_NSAMPLES = 200


# ==================== 第 3 步：读取并检查模型 ====================
@st.cache_resource(show_spinner="Loading SVM model...")
def load_model():
    """只在首次运行时读取模型，之后网页刷新会复用缓存中的模型。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{MODEL_PATH.name}。请把它放到 app.py 同一目录。"
        )

    # pickle 文件可以执行其中携带的对象代码，因此只能加载自己训练或可信来源的文件。
    with MODEL_PATH.open("rb") as file:
        loaded_model = pickle.load(file)

    # 本应用需要概率，而不是只有 0/1 的类别结果。
    # 用户给出的 SVC(probability=True) 具备 predict_proba 方法。
    if not hasattr(loaded_model, "predict_proba"):
        raise TypeError(
            "当前模型没有 predict_proba 方法。训练 SVC 时必须设置 probability=True。"
        )

    if not hasattr(loaded_model, "classes_"):
        raise TypeError("当前模型没有 classes_ 属性，无法识别 Outcome=1 的概率列。")

    return loaded_model


# ==================== 第 4 步：读取并检查 SHAP 汇总背景数据 ====================
@st.cache_data(show_spinner="Loading SHAP background data...")
def load_background_features():
    """读取不含患者原始记录的汇总背景数据，并保持固定列顺序。"""
    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(
            f"找不到 SHAP 背景数据：{BACKGROUND_PATH.name}。请把它放到 app.py 同一目录。"
        )

    # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8 CSV。
    data = pd.read_csv(BACKGROUND_PATH, encoding="utf-8-sig")

    # 去掉列名前后的意外空格或制表符，例如 "Age\t" 会被清理为 "Age"。
    data.columns = data.columns.astype(str).str.strip()

    missing_columns = [name for name in FEATURE_NAMES if name not in data.columns]
    if missing_columns:
        raise ValueError(
            "shap_background.csv 缺少以下特征列：" + ", ".join(missing_columns)
        )

    # Outcome 和其他变量不能进入模型；这里只按固定顺序提取 9 个特征。
    features = data.loc[:, FEATURE_NAMES].copy()

    # 强制转换为数字。如果 CSV 中混入文字，会在这里给出明确错误，而不是静默预测。
    for column in FEATURE_NAMES:
        features[column] = pd.to_numeric(features[column], errors="raise")

    if features.isna().any().any():
        raise ValueError("9 个模型特征中存在缺失值，请先处理缺失值再部署。")

    if features.empty:
        raise ValueError("shap_background.csv 没有数据行，无法建立 SHAP 解释器。")

    return features


# ==================== 第 5 步：确定 Outcome=1 对应哪一列概率 ====================
def get_positive_class_index(model):
    """返回 predict_proba 输出中 Outcome=1 所在的列下标。"""
    classes = list(model.classes_)

    # 常见情况：Outcome 是整数 0/1。
    if 1 in classes:
        return classes.index(1)

    # 某些 CSV 会把 Outcome 保存为字符串 "0"/"1"。
    if "1" in classes:
        return classes.index("1")

    raise ValueError(
        f"模型类别为 {classes}，其中没有 1。请确认 Outcome=1 表示显著肝纤维化。"
    )


# ==================== 第 6 步：创建 SHAP 解释器 ====================
@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def create_shap_explainer():
    """创建一次 KernelExplainer，并在后续预测中复用。"""
    model = load_model()
    background_features = load_background_features()
    positive_class_index = get_positive_class_index(model)

    # 公开部署只加载聚类中心等汇总值，不上传完整患者级训练数据。
    background = background_features.to_numpy(dtype=float)

    def predict_positive_probability(values):
        """把 SHAP 给出的数组还原为 DataFrame，并只返回 Outcome=1 的概率。"""
        values_df = pd.DataFrame(values, columns=FEATURE_NAMES)
        probabilities = model.predict_proba(values_df)
        return probabilities[:, positive_class_index]

    return shap.KernelExplainer(predict_positive_probability, background)


# ==================== 第 7 步：绘制单个病例的 SHAP force plot ====================
def draw_shap_force_plot(input_data):
    """计算当前一行数据的 SHAP 值，并返回 Matplotlib 图。"""
    explainer = create_shap_explainer()

    # KernelExplainer 接收二维数组；即使只有一个病例也要保持形状为 (1, 9)。
    shap_values = explainer.shap_values(
        input_data.to_numpy(dtype=float),
        nsamples=SHAP_NSAMPLES,
        silent=True,
    )

    # 不同 SHAP 版本的返回外层结构可能略有差异，这里统一整理成 9 个数值。
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    values = np.asarray(shap_values, dtype=float).reshape(-1)

    if values.size != len(FEATURE_NAMES):
        raise ValueError(
            f"SHAP 返回了 {values.size} 个解释值，但应用需要 {len(FEATURE_NAMES)} 个。"
        )

    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    # matplotlib=True 会生成静态图片，能直接显示在 Streamlit Community Cloud 中。
    plt.figure(figsize=(14, 3.2))
    shap.force_plot(
        base_value,
        values,
        input_data.iloc[0],
        feature_names=FEATURE_NAMES,
        matplotlib=True,
        show=False,
        contribution_threshold=0.01,
    )
    figure = plt.gcf()
    figure.tight_layout()
    return figure


# ==================== 第 8 步：创建网页标题、说明和输入框 ====================
st.title("Prediction Model with SHAP Visualization")
st.write(
    "This app predicts the possibility of Significant Fibrosis with a trained "
    "SVM model and explains the individual prediction using SHAP."
)
st.subheader("Enter the following feature values:")

# 用字典收集用户输入。三列布局在电脑端更紧凑，在手机端会自动换行。
user_values = {}
columns = st.columns(3)

for index, feature_name in enumerate(FEATURE_NAMES):
    config = INPUT_CONFIG[feature_name]
    with columns[index % 3]:
        user_values[feature_name] = st.number_input(
            label=(
                f"{feature_name} "
                f"({config['min']:g} - {config['max']:g})"
            ),
            min_value=float(config["min"]),
            max_value=float(config["max"]),
            value=float(config["default"]),
            step=float(config["step"]),
            format="%.2f",
            key=feature_name,
        )


# ==================== 第 9 步：点击 Predict 后预测并显示 SHAP 图 ====================
if st.button("Predict", type="primary", use_container_width=True):
    try:
        model = load_model()
        positive_class_index = get_positive_class_index(model)

        # 用 FEATURE_NAMES 再次指定列顺序，确保模型收到的顺序与训练时完全一致。
        input_data = pd.DataFrame(
            [[user_values[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )

        predicted_probability = float(
            model.predict_proba(input_data)[0, positive_class_index]
        )

        # .2% 会把 0.8616 自动显示成 86.16%。
        st.success(
            "Based on feature values, predicted possibility of "
            f"Significant Fibrosis is **{predicted_probability:.2%}**"
        )

        st.subheader("SHAP explanation for this prediction")
        with st.spinner("Calculating SHAP values..."):
            shap_figure = draw_shap_force_plot(input_data)
            st.pyplot(shap_figure, use_container_width=True)
            plt.close(shap_figure)

        st.caption(
            "Red features push the prediction toward a higher probability; "
            "blue features push it toward a lower probability."
        )

    except Exception as error:
        # 网页上线后如果文件、版本或列名有问题，会直接显示可读错误，便于排查。
        st.error(f"Prediction failed: {error}")
        st.exception(error)


# ==================== 第 10 步：显示用途说明 ====================
st.divider()
st.caption(
    "For research and educational use only. This result is not a medical diagnosis "
    "and must not replace evaluation by qualified healthcare professionals."
)
