"""
utils.py - plotting and load/save helpers
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_label_distribution(labels, title="Label distribution"):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8,4))
    plt.bar(unique, counts)
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
