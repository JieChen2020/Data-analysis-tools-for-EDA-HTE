from pandas import read_csv
from chemplot import Plotter
import matplotlib.pyplot as plt


data_BBBP = read_csv("acid_sampled_file.csv")
# data_BBBP = read_csv("amine_sampled_file.csv")
cp = Plotter.from_smiles(data_BBBP["smiles"], target=data_BBBP["target"], target_type="C", sim_type="structural")
a = cp.tsne(pca=False, random_state=20)
a.to_csv('acid_TSNE.csv', index=False)
# a.to_csv('amine_TSNE.csv', index=False)
cp.visualize_plot()
plt.show()

