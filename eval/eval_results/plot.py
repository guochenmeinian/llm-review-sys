import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eval/eval_results/summary.csv")

plt.style.use("ggplot")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 字体大小统一
label_fontsize = 10
tick_fontsize = 10

# OQ MAE
df.plot.bar(x="Model", y="OQ_MAE", ax=axes[0][0], legend=False)
axes[0][0].set_title("Overall Quality MAE")
axes[0][0].tick_params(axis='x', labelrotation=30, labelsize=tick_fontsize)

# OQ RMSE
df.plot.bar(x="Model", y="OQ_RMSE", ax=axes[0][1], legend=False)
axes[0][1].set_title("Overall Quality RMSE")
axes[0][1].tick_params(axis='x', labelrotation=30, labelsize=tick_fontsize)

# OQ Pearson
df.plot.bar(x="Model", y="OQ_Pearson", ax=axes[1][0], legend=False)
axes[1][0].set_title("Overall Quality Pearson")
axes[1][0].tick_params(axis='x', labelrotation=30, labelsize=tick_fontsize)

# BLEU + ROUGE
df.plot.bar(x="Model", y=["BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"], ax=axes[1][1])
axes[1][1].set_title("BLEU-4 / ROUGE Scores")
axes[1][1].tick_params(axis='x', labelrotation=30, labelsize=tick_fontsize)
axes[1][1].legend(fontsize=label_fontsize)

plt.tight_layout()
plt.show()