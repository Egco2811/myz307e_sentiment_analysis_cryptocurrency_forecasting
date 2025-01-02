"""
enhanced_error_plots.py

Purpose:
- Create box/violin plots of error distributions across multiple folds or time slices.
- Visualize not just the mean error but also variance, outliers, etc.

Usage:
1) Prepare a dictionary like:
   errors_dict = {
       "ARIMA": [err_fold1, err_fold2, ...],
       "LSTM-only": [...],
       "Lexicon-LSTM": [...],
       "BERT-LSTM": [...]
   }
   Where each value is a list/array of error metrics (e.g., absolute errors) for each fold.

2) Call plot_error_distributions(errors_dict, plot_type="box" or "violin").

Implementation Approach:
- We create a long-form data structure with columns: [Model, Error, FoldIndex].
- Then we plot either a boxplot or a violinplot using seaborn or matplotlib.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_error_distributions(errors_dict: dict,
                             plot_type: str = "box",
                             out_file: str = "error_distribution_plot.png"):
    """
    Generate box or violin plots comparing error distributions for each model.

    Parameters
    ----------
    errors_dict : dict
        Dictionary of {model_name: [list_of_errors_across_folds]}.
        Example:
            {
              "ARIMA": [0.1, 0.2, 0.05, ...],
              "LSTM-only": [...],
              "Lexicon-LSTM": [...],
              "BERT-LSTM": [...]
            }
        Each value can be a list/array of error values (e.g., absolute errors) aggregated across folds.
    plot_type : str
        "box" for boxplot, "violin" for violin plot.
    out_file : str
        Filename for saving the figure.

    Returns
    -------
    None
    """
    # 1) Convert the dictionary into a long-form DataFrame
    #    with columns: Model, Error
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
