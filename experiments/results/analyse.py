import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

if __name__ == "__main__":
    sns.set(font_scale=1.5)
    # Iterate over the directories in ./results
    for dir_name in os.listdir("."):
        dir_path = os.path.join(".", dir_name)
        # Check if it is a directory

        if dir_name == "weak_arrow":
            const = 1.010449
        elif dir_name == "large_backdoor":
            const = 0.902034
        else :
            continue
            const = 0.9

        if os.path.isdir(dir_path):
            # Iterate over the files in the directory
            mat = pd.read_csv(os.path.join(dir_path, "result_matrix.csv"), index_col=0)
            univ_mat = pd.read_csv(os.path.join(dir_path, "univariate_matrix.csv"), index_col=0)

            normalized_df = np.abs(mat / const)

            # Rename columns, format "n interventions"
            normalized_df.columns = [str(i+1) for i in range(len(normalized_df.columns))]

            # Rename index, format "n observations"
            normalized_df.index = [str(i+1) for i in range(len(normalized_df.index))]

            # Drop na columns
            normalized_df = normalized_df.dropna(axis=1, how='all')


            # Create the heatmap
            plt.figure(figsize=(10, 8))  # Adjust size as needed
            heatmap = sns.heatmap(normalized_df,
                                  annot=True,  # Shows values in cells
                                  fmt=".2f",  # Format with 1 decimal place
                                  cmap='viridis',  # Choose a colormap
                                  vmin=0,  # Minimum value for color scale
                                  vmax=1)  # Maximum value for color scale

            plt.title("Counterfactual vs. semi-counterfactual excess variance")
            plt.xlabel("Number of Interventions")
            plt.ylabel("Number of Observations")
            # Invert y-axis to have the origin at the bottom-left
            plt.gca().invert_yaxis()
            plt.tight_layout()
            #plt.show()
            plt.savefig(os.path.join(dir_path, "heatmap_multi_"+ str(dir_name) +".pdf"))

            # Create the univariate heatmap
            normalized_df = np.abs(univ_mat / const)

            # Rename columns, format "n interventions"
            normalized_df.columns = ["X"+str(i) for i in range(len(normalized_df.columns))]

            # Rename index, format "n observations"
            normalized_df.index = ["X"+str(i) for i in range(len(normalized_df.index))]

            # Threshold the values to 100
            normalized_df[normalized_df > 1] = 1
            plt.figure(figsize=(10, 8))
            heatmap = sns.heatmap(normalized_df,
                                  annot=True,  # Shows values in cells
                                  fmt=".2f",  # Format with 1 decimal place
                                  cmap='viridis',  # Choose a colormap
                                  vmin=0,  # Minimum value for color scale
                                  vmax=1)
            plt.title("Single node semi-counterfactual excess variance")
            plt.xlabel("Intervened node")
            plt.ylabel("Observed node")
            plt.tight_layout()
            #plt.show()
            plt.savefig(os.path.join(dir_path, "heatmap_univariate_" + str(dir_name) + ".pdf"))



