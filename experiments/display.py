import pandas as pd
import matplotlib.pyplot as plt

fname = 'side_length_ratios.csv'

df = pd.read_csv(fname)

data = df[['Length ratio', 'Count ratio']].sort_values(by='Length ratio').to_numpy()

plt.plot(data[:, 0], data[:, 1])
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.xlabel('Side length ratio')
plt.ylabel('Count ratio')
plt.show()