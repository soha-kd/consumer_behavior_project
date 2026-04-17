"""
Utility module containing helper functions that are reused across multiple files
in the project.
"""
import time
import os
import matplotlib.pyplot as plt

def wrapper_for_fit(fit_func, X_train, y_train, desc):
    print(f"\n\nTraining {desc}...")
    start = time.time()
    result = fit_func(X_train, y_train)
    end = time.time()
    print(f"{desc} finished in {end - start:.2f} seconds\n")

    return result


def save_plot(filename, folder="outputs/plots", show=True):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    plt.savefig(filepath, bbox_inches="tight")
    print(f"Saved plot: {filepath}")

    #if show:
      #  plt.show()

    plt.close("all")  