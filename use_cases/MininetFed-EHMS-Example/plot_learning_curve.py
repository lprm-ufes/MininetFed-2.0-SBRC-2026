import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

df = pd.read_csv("server_output/learning_curve.csv")

fig, ax = plt.subplots(figsize=(3.0, 1.2))

ax.plot(
    df["round"],
    df["f1"],
    marker="o",
    markersize=2.2,
    linewidth=1.0,
)

ax.set_xlabel("Rodada", labelpad=1)
ax.set_ylabel("F1-Score", labelpad=1)

ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(pad=0.2)
plt.savefig("ehms_lc_non_iid.pdf", bbox_inches="tight", pad_inches=0.01)
plt.show()
