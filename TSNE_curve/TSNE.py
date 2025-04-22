from pandas import read_csv
from chemplot import Plotter
import matplotlib.pyplot as plt


data_BBBP = read_csv("filtered_molecules.csv")
cp = Plotter.from_smiles(data_BBBP["smiles"], target=data_BBBP["target"], target_type="C", sim_type="structural")
a = cp.tsne(pca=False, random_state=20)
a.to_csv('TSNE.csv', index=False)
cp.visualize_plot()
plt.show()
