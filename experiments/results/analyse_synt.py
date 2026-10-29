import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set global seaborn styling
sns.set_theme(style="whitegrid")
num_ints = "# interventions"  # "Number of interventions"
num_obs = "# observations" # "Number of observations"


def plot_segregated_analysis(df, p_val, root_dir, metric = 'median_wass'):
    plot_dir = os.path.join(root_dir, "plots", metric)

    # Create the directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    """Generates and saves individual plots for a specific p-value."""
    print(f"Generating segregated plots for p = {p_val}...")

    # 1. Heatmap: num_obs vs num_ints (Values: median_wass)
    plt.figure(figsize=(10/np.sqrt(2), 8/np.sqrt(2)))
    # Using pivot_table handles potential duplicate entries by averaging them
    heatmap_data = df.pivot_table(index='num_obs', columns='num_ints', values=metric, aggfunc='mean')
    if metric == 'median_wass':
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
                    #cbar_kws={'label': 'Median wasserstein distance'})

        plt.title(f'Median Wasserstein distance ($\epsilon$={p_val})')
    else :
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
                    #cbar_kws={'label': 'Mean wasserstein distance'})
        plt.title(f'Mean Wasserstein distance ($\epsilon$={p_val})')
    plt.xlabel(num_ints)
    plt.ylabel(num_obs)
    plt.gca().invert_yaxis()  # Standardize origin to bottom-left if desired
    plt.tight_layout()
    plot_file_name = os.path.join(plot_dir, f'heatmap_p_{p_val}.pdf')
    plt.savefig(plot_file_name, dpi=300)
    plt.close()

    # 2. Time vs Accuracy Trade-off Scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x='time_seconds', y=metric,
        hue='num_ints', size='num_obs', palette='viridis',
        sizes=(40, 300), alpha=0.8
    )
    plt.title(f'Time vs accuracy trade-off (p={p_val})')
    if metric == 'median_wass':
        plt.ylabel('Median Wasserstein distance')
    else :
        plt.ylabel('Mean Wasserstein distance')
    plt.xlabel('Execution time (seconds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_file_name = os.path.join(plot_dir, f'tradeoff_p_{p_val}.pdf')
    plt.savefig(plot_file_name, dpi=300)
    plt.close()


def plot_global_analysis(combined_df, root_dir, metric = 'median_wass'):
    plot_dir = os.path.join(root_dir, "plots")

    # Create the directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    """Generates plots analyzing trends across all p-values."""
    print("Generating global cross-p plots...")

    # Inside plot_global_analysis(combined_df):

    # 1. Convert to string to force categorical treatment
    combined_df = combined_df.copy()  # Avoid modifying original dataframe
    combined_df['num_obs_str'] = combined_df['num_obs'].astype(str)

    # Rename columns on the fly so Seaborn reads the nice names
    plot_df = combined_df.rename(columns={'num_ints': num_ints, 'num_obs_str': num_obs})

    # 2. Sort numerically so the legend and color gradient stay in the correct order
    ordered_obs = [str(x) for x in sorted(combined_df['num_obs'].unique())]

    plt.figure(figsize=(10/np.sqrt(2), 8/np.sqrt(2)))
    sns.scatterplot(
        data=plot_df,
        x='p',
        y=metric,
        hue=num_obs,  # Color scales with num_obs
        hue_order = ordered_obs,  # Ensure correct order in legend
        size=num_ints,  # Bubble size scales with num_ints
        palette='viridis',  # Sequential colormap
        sizes=(20, 300),  # Min and max bubble sizes
        alpha=0.7,  # Transparency to see overlapping points
        edgecolor='black',
        linewidth=0.5
    )

    plt.title('Global sensitivity to $\epsilon$') # \n(Color = Observations, Size = Interventions)')
    plt.xlabel('Parameter $\epsilon$')
    plt.ylabel('Mean Wasserstein distance')

    # Adjust legend so it doesn't overlap the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plot_file_path = os.path.join(plot_dir, f'{metric}_global_single_bubble_plot.pdf')
    plt.savefig(plot_file_path, dpi=300)
    plt.close()

    # 1. Sensitivity to p (Aggregating across num_obs and num_ints)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=combined_df, x='p', y=metric,
        hue='num_ints', palette='coolwarm', marker='s', err_style="bars"
    )
    plt.title('Sensitivity to $\epsilon$')
    plt.ylabel('Mean Wasserstein distance')
    plt.xlabel('Parameter $\epsilon$')
    plt.legend(title=num_ints, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_file_path = os.path.join(plot_dir, f'{metric}_global_sensitivity_p.pdf')
    plt.savefig(plot_file_path, dpi=300)
    plt.close()



def main(root_dir = os.path.join("synthetic", "nodes_8")):
    # Replace this dictionary with your actual dictionary!
    # Keys can be anything (like the p-value), values must be the file paths

    all_data = []

    # Iterate over the dictionary instead of using glob
    for file_name in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file_name)
        if not file_path.endswith(".csv"):
            continue
        df = pd.read_csv(file_path)

        # Ensure we have a valid dataframe
        if df.empty:
            print(f"File '{file_path}' is empty. Skipping...")
            continue

        # Extract 'p' value (Assuming 'p' is a column, we take the first value)
        # It guarantees the plot title uses the exact numeric 'p' from the data
        p_val = df['p'].iloc[0]

        # Generate individual segregated plots
        plot_segregated_analysis(df, p_val, root_dir, metric='median_wass')
        plot_segregated_analysis(df, p_val, root_dir, metric='mean_wass')

        # Store for global analysis
        all_data.append(df)

    # Combine all data for global analysis
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        plot_global_analysis(combined_df, root_dir, metric='median_wass')
        plot_global_analysis(combined_df, root_dir, metric='mean_wass')
        print("All analyses completed successfully! Check your directory for the .pdf files.")
    else:
        print("No data was loaded. Please check the file paths in your dictionary.")


if __name__ == "__main__":
    main()