import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('logs/eval.csv', index_col=0)
df.plot(x=df.index, y=df.columns[0])
plt.title("Generator Inception Score during training")
plt.show()

df.plot(x=df.index, y=df.columns[1])
plt.title("Linear MMD Statistic during training")
plt.show()
