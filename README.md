# Significant Fibrosis Prediction with SHAP

这是一个基于 Streamlit 的二分类预测网页应用。应用加载已训练的 SVM 模型，预测 `Outcome=1`（显著肝纤维化）的概率，并使用 SHAP 解释单个预测结果。

## 在线部署文件

- `app.py`：Streamlit 应用。
- `svm_model.pkl`：训练完成的 SVM 模型。
- `shap_background.csv`：仅包含 20 行聚类中心的 SHAP 汇总背景数据。
- `requirements.txt`：Python 依赖版本。

完整患者级训练数据 `traindata.csv` 已通过 `.gitignore` 排除，不会上传到公开仓库。

## 本地运行

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

> 本项目仅供科研与教学使用，不能替代专业医务人员的诊断。
