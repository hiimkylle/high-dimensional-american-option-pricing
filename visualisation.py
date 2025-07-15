import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def joint_structure_plots(X_np, Y_model, sample=1200):
    d = X_np.shape[1]
    idx = np.random.choice(len(X_np), min(sample, len(X_np)), replace=False)
    idxm = np.random.choice(len(Y_model), min(sample, len(Y_model)), replace=False)
    df_emp = pd.DataFrame(X_np[idx])
    df_emp["Type"] = "Empirical"
    df_mod = pd.DataFrame(Y_model[idxm])
    df_mod["Type"] = "Model"
    df = pd.concat([df_emp, df_mod], ignore_index=True)

    sns.pairplot(df, hue="Type", plot_kws=dict(alpha=0.25, s=12))
    plt.suptitle("Pairwise Marginals and Joint Structure: Empirical vs Model", y=1.02)
    plt.show()
