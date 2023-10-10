"""
Analysis script for RP moral weight correlation analysis.

Generates overall correlation matrix and example species-to-species heatmaps.
Then runs a pair of example analyses.

"""

# Imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import rpwelfarecorr as rwq
import pandas as pd

# Species list (with names used in RP analysis)
species_list = ['pigs', 'chickens', 'octopuses', 'carp', 'bees', 'salmon',
                'crayfish', 'shrimp', 'crabs', 'bsf', 'silkworms']

# Species list (with capitalization for plot labeling)
species_list_caps = ['Pigs', 'Chickens', 'Octopuses', 'Carp', 'Bees', 'Salmon',
                     'Crayfish', 'Shrimp', 'Crabs', 'BSF', 'Silkworms']

# Number of samples to use
num_samples = 10000

# Plot setup
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
mpl.rcParams['figure.dpi'] = 300
num_bins = 100
my_pal = {"lightsteelblue", "lightcoral", "thistle", "navajowhite"}

# Sample from distribution for each welfare model
data_per_model = rwq.gen_corr_samples(species_list, num_samples)

# Compute summary statistics for a few species (allows for checking)
rwq.compute_summary_stats_arr(data_per_model['pigs'], print_en=True,
                              name="Pigs - Paired from Component")
rwq.compute_summary_stats_arr(data_per_model['chickens'], print_en=True,
                              name="Chickens - Paired from Component")
rwq.compute_summary_stats_arr(data_per_model['bsf'], print_en=True,
                              name="Black Soldier Flies - Paired from Component")

# Create example heatmpas for pigs vs. chickens and BSF vs. silkworms
rwq.heatmap_wr_ranges(data_per_model['pigs'], data_per_model['chickens'],
                      'Pigs', 'Chickens',
                      'Pigs vs. Chickens, Paired Samplig',
                      num_bins=num_bins, save_en=False)

num_bins = 200
rwq.heatmap_wr_ranges(data_per_model['bsf'], data_per_model['silkworms'],
                      'Black Soldier Flies', 'SilkWorms',
                      'BSF vs. Silkworms, Paired Samplig',
                      num_bins=num_bins, save_en=False,lims=[0, 0.5])

df = pd.DataFrame(data_per_model, columns=species_list)
df.columns = species_list_caps

correlation_mat = df.corr()
sns.heatmap(correlation_mat, annot = True,cmap=mpl.cm.Reds)
plt.title("Correlations for Rethink Priorities Moral Weights\nUsing Pair-Wise Sampling from Constituent Models")
plt.show()
