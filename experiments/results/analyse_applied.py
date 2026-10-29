from itertools import product, combinations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming these are imported from your local module
from experiments.main_applied import variables, SHMIN, SHMAX, PP

# ---------------------------------------------------------
# FIRST TABLE & HEATMAP
# ---------------------------------------------------------
# Create df with arange(36,40,0.25) index and variables columns
df_final = pd.DataFrame(index=np.arange(36, 40, 0.25), columns=variables[:8])

for int_value in df_final.index:
    for i, var in enumerate(df_final.columns):
        # Load file
        file_path = f"./applied/samples/{i}_{int_value}.csv"
        df_samples = pd.read_csv(file_path)

        # Compute absolute difference between the two columns
        tv = np.abs((df_samples["SCM"].mean() - df_samples["CBN"].mean()))
        df_final.loc[int_value, var] = tv

# Threshold at 1e-2 to avoid printing very small values
df_final = df_final.applymap(lambda x: x if abs(x) > 1e-2 else 0)
df_final = df_final.astype(float)  # Ensure numeric type for Seaborn

# -> FORMAT ROW LABELS TO INCLUDE .0 <-
# We do this after loading the data so the file paths aren't affected
df_final.index = [f"{float(x):.1f}" if float(x).is_integer() else str(x) for x in df_final.index]

# Transpoase the DataFrame for better visualization in the heatmap
#df_final = df_final.T

print("--- First DataFrame ---")
print(df_final.to_string())

# Plot First Heatmap
plt.figure(figsize=(10/np.sqrt(1.4), 8/np.sqrt(1.4)))
sns.heatmap(df_final, annot=True, cmap="viridis", fmt=".2f", linewidths=.5)

# -> INCREASE FONT SIZES <-
#plt.title("Difference in predicted distribution", fontsize=18)
plt.ylabel("Pore pressure intervention (MPa)", fontsize=14)
plt.xlabel("Variables observed (cumulative)", fontsize=14)

plt.xticks(rotation=40) # Make sure row labels are completely horizontal
plt.tight_layout()
plt.savefig("./applied/diff.pdf")
plt.show()

# ---------------------------------------------------------
# SECOND TABLE & HEATMAP
# ---------------------------------------------------------
child = [PP, SHMIN, SHMAX]
combinations_list = list(combinations(child, r=1)) + list(combinations(child, r=2)) + list(combinations(child, r=3))

df_final2 = pd.DataFrame(index=np.arange(36, 40, 0.25), columns=combinations_list)

for i, int_value in enumerate(df_final2.index):
    for j, _ in enumerate(df_final2.columns):
        # Load file
        file_path = f"./applied/samples_children/{j}_{int_value}.csv"
        df_samples = pd.read_csv(file_path)

        # Compute absolute difference between the two columns
        tv = np.abs((df_samples["SCM"].mean() - df_samples["CBN"].mean()))
        df_final2.iloc[i, j] = tv

df_final2 = df_final2.applymap(lambda x: x if abs(x) > 1e-2 else 0)
df_final2 = df_final2.astype(float)  # Ensure numeric type for Seaborn

# Format column names with newline character
df_final2.columns = ["\n".join(col) for col in df_final2.columns]

# -> FORMAT ROW LABELS TO INCLUDE .0 <-
df_final2.index = [f"{float(x):.1f}" if float(x).is_integer() else str(x) for x in df_final2.index]

#df_final2 = df_final2.T

print("\n--- Second DataFrame ---")
print(df_final2.to_string())

# Plot Second Heatmap
plt.figure(figsize=(10/np.sqrt(1.4), 8/np.sqrt(1.4)))
sns.heatmap(df_final2, annot=True, cmap="viridis", fmt=".2f", linewidths=.5)

# -> INCREASE FONT SIZES <-
#plt.title("Difference in predicted distribution", fontsize=18)
plt.ylabel("Pore pressure intervention (MPa)", fontsize=14)
plt.xlabel("Variables observed", fontsize=14)

plt.xticks(rotation=60, ha='center')
plt.yticks(rotation=0) # Make sure row labels are completely horizontal
plt.tight_layout()
plt.savefig("./applied/diff_children.pdf", dpi=300)
plt.show()
