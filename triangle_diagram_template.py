import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern

data_I = pd.read_csv("Phase_1_data.csv").to_numpy()[1:]
data_II = pd.read_csv("Phase_2_data.csv").to_numpy()[1:]

# Multiply by 100 ONLY ONCE to match the 0-100 scale on the axes
data_I = data_I * 100
data_II = data_II * 100

benz_I, acet_I, wat_I = data_I[:, 0], data_I[:, 1], data_I[:, 2]
benz_II, acet_II, wat_II = data_II[:, 0], data_II[:, 1], data_II[:, 2]

# Adjusted figsize for better proportions
fig = plt.figure(figsize=(12.5, 12.5))
ax = fig.add_subplot(projection="ternary", ternary_sum=100.0)

# Increased font sizes
ax.set_tlabel("Acetone", fontsize=18)
ax.set_llabel("Benzene", fontsize=18)
ax.set_rlabel("Water", fontsize=18)
ax.tick_params(labelsize=16) # Makes the 0, 20, 40... numbers larger

Acet_curve = np.concatenate([acet_I, acet_II[::-1]])
Benz_curve = np.concatenate([benz_I, benz_II[::-1]])
Wat_curve = np.concatenate([wat_I, wat_II[::-1]])

# Fixed the typo here
ax.plot(Acet_curve, Benz_curve, Wat_curve, color="tab:blue", label="UNIQUAC calculated")

plait_Acet, plait_Benz, plait_Wat = acet_I[-1], benz_I[-1], wat_I[-1]
ax.scatter(plait_Acet, plait_Benz, plait_Wat, marker="x", color="red", s=50, label="plait point")

tie_lines = np.linspace(0, len(acet_I) - 1, 15, dtype=int)

for idx, i in enumerate(tie_lines):
    label = "tie-lines" if idx == 0 else None
    ax.plot([acet_I[i], acet_II[i]],
            [benz_I[i], benz_II[i]],
            [wat_I[i], wat_II[i]],
            color="grey", lw=0.8, label=label)

ax.plot([0, 0], [100, 0], [0, 100], color="grey", lw=0.8)
ax.grid()

# Increased legend font size
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False, fontsize=14)
plt.figtext(0.5, 0.01, "Figure 1: Ternary diagram for the system", ha="center", fontsize=12)

# Prevent text cutoff on save
plt.subplots_adjust(left=0.15, right=0.75, bottom=0.15)
plt.savefig("LLE_Diagram.png", dpi=300, bbox_inches="tight", pad_inches=0.5)
plt.show()