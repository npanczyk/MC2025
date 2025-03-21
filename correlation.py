import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

''' Get correlation matrix '''

# Read and label data
df = pd.read_csv("chf_train_synth.csv")

df = df.apply(pd.to_numeric)

correlation_matrix = df.corr()
#print(correlation_matrix)

# plot
plt.figure(figsize=(6, 5))
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", dpi=300) # change path as necessary