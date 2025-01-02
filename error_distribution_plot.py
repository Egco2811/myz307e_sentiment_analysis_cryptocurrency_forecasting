import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_error_distributions(errors_dict: dict,
                             plot_type: str = "box",
                             out_file: str = "error_distribution_plot.png"):

    # 1) Convert the dictionary into a long-form DataFrame
    
    rows = []
    for model_name, err_list in errors_dict.items():
        for error_val in err_list:
            rows.append([model_name, error_val])
    df = pd.DataFrame(rows, columns=["Model", "Error"])

    # 2) Plot
    plt.figure(figsize=(8, 6))
    
    if plot_type.lower() == "violin":
        sns.violinplot(x="Model", y="Error", data=df, cut=0, inner="box", palette="Set2")
        plt.title("Violin Plot of Error Distributions")
    else:
        sns.boxplot(x="Model", y="Error", data=df, palette="Set2")
        plt.title("Box Plot of Error Distributions")
    
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"{plot_type.capitalize()} plot saved to {out_file}")
