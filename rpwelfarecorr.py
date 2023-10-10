"""
Correlation analysis tools for Rethink Priorities p(sentience)-adjusted
welfare range estimates

Functions:

    gen_corr_samples(species_list, num_samples)
    adj_wr_correlation(species, species_wr, num_samples)
    heatmap_wr_ranges(data1, data2, animal1, animal2, title_str,
                      num_bins, save_en, lims)
    compute_summary_stats_arr(samples_arr, print_en, name):

Misc variables:

    wr_est_dir
    sent_est_dir

See forum.effectivealtruism.org/posts/Qk3hd6PrFManj8K6o

"""

# Imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import random
import math

# Setup
wr_est_dir = os.path.join(os.path.dirname(__file__), 'WR_Outputs')
sent_est_dir = os.path.join(os.path.dirname(__file__), 'Sentience_Estimates')


def gen_corr_samples(species_list, num_samples):
    """
    Generate coorelated samples from the RP moral weight distributions.

        Parameters_
            species_list:    List of species names to generate samples
                             for. Allowable values:
                              - 'pigs'
                              -  'chickens'
                              - 'octopuses'
                              - 'carp'
                              - 'bees'
                              - 'salmon'
                              - 'crayfish'
                              - 'shrimp'
                              - 'crabs'
                              - 'bsf'
                              - 'silkworms'
            num_samples:     Number of samples to generate (int)

        Returns_
            corr_samples:    Dictionary containing a list of num_samples
                             draws per species in species_list

    """
    # Warn user if using a small number of samples
    if num_samples < 90:
        warnings.warn('Low number of samples requested. Medians may be skewed')

    # Models in the RP mixture model
    models = ['Qualitative', 'High-Confidence (Simple Scoring)',
              'Cubic', 'High-Confidence (Cubic)',
              'Qualitative Minus Social', 'Pleasure-and-pain-centric',
              'Higher-Lower Pleasures', 'Undiluted Experience', 'Neuron Count']
    num_models = len(models)

    # Declarations
    num_models = len(models)
    data_unadj = {}
    corr_samples = {}

    # For each species, load the unadjusted welfare range simulation
    # data for each welfare model
    for species in species_list:
        for model in models:
            data_unadj[species, model] = pickle.load(open(os.path.join(
                wr_est_dir, '{} {}.p'.format(species, model)), 'rb'))

    # Determine how many draws to pull from each constituent model
    samp_per_model = math.floor(num_samples/num_models)
    extra_samp_num = num_samples % num_models
    draw_nums = []
    draw_nums.extend([samp_per_model]*(9-extra_samp_num))
    draw_nums.extend([samp_per_model+1]*extra_samp_num)
    random.shuffle(draw_nums)
    print(draw_nums)

    # Sample from distribution for each welfare model and adjust for
    # p(sentience)
    for species in species_list:
        corr_samples[species] = []
        for i in range(0, num_models):
            model = models[i]
            corr_samples[species].extend(
                np.random.choice(data_unadj[species, model], draw_nums[i]))

        corr_samples[species] = \
            adj_wr_correlation(species, corr_samples[species], num_samples)

    # Return dictionary of samples
    return corr_samples


def adj_wr_correlation(species, species_wr, num_samples):
    """
    Adjust a welfare range estimate for p(sentience).

        Parameters_
            species:     Species string. Allowable values:
                          - 'pigs'
                          -  'chickens'
                          - 'octopuses'
                          - 'carp'
                          - 'bees'
                          - 'salmon'
                          - 'crayfish'
                          - 'shrimp'
                          - 'crabs'
                          - 'bsf'
                          - 'silkworms'
            species_wr   List of samples for species welfare range
            num_samples: Number of samples to generate (int)

        Returns_
            species_adj_wr: List of adjusted welfare range estimate samples

        Notes_
            Created based on one_species_adj_wr in RP moral weights
            project code
    """
    # Need to treat shrimp specilly since they were not included in the
    # p(sentience) calculations
    if species != 'shrimp':
        with open(os.path.join(sent_est_dir,
                  '{}_psent_hv1_model.p'.format(species)), 'rb') as f_s:
            species_psent = list(pickle.load(f_s))
    else:
        with open(os.path.join(sent_est_dir,
                  'shrimp_assumed_psent.p'), 'rb') as f_s:
            species_psent = list(pickle.load(f_s))

    species_adj_wr = []

    # Multiply each welfare range estimate by each p(sentience) estimate
    for i in range(num_samples):
        psent_i = species_psent[i]
        wr_i = species_wr[i]
        adj_wr_i = max(psent_i*wr_i, 0)
        species_adj_wr.append(adj_wr_i)

    return species_adj_wr


def heatmap_wr_ranges(data1, data2, animal1, animal2, title_str,
                      num_bins=20, save_en=False, lims=[0, 2]):
    """
    Create a heatmap for two welfare ranges.

        Parameters_
            data1:           Welfare range estimate data for 1st species
            data2:           Welfare range estimate data for 2nd species
            animal1:         Name of 1st species
            animal1:         Name of 2nd species
            title_str:       Figure title
            num_bins:        Number of bins per axis for heatmap
            save_en:         Enable saving of figure
            lims:            X/Y plot limits

    """
    # Compute correlation and mean data for labeling
    r = np.corrcoef(data1, data2)
    mean1 = round(np.mean(data1), 4)
    mean2 = round(np.mean(data2), 4)
    correlation_coeff = round(r[0, 1], 4)

    # Create figure and plot heatmap
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('lightgray')
    h = plt.hist2d(data1, data2, density=True, bins=num_bins,
                   norm=mpl.colors.LogNorm(), cmap=mpl.cm.Reds)

    # Compute text location
    text_loc = [lims[1]*0.525, lims[1]*0.9]

    # Add labels
    ax.set_xlabel('P(sentience)-Adjusted Welfare Range of {}'.format(animal1))
    ax.set_ylabel('P(sentience)-Adjusted Welfare Range of {}'.format(animal2))
    plt.text(text_loc[0], text_loc[1],
             'Mean ({}) = {} \nMean ({}) = {} \nCorrelation = {}'
             .format(animal1, mean1, animal2, mean2, correlation_coeff))
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Density', rotation=270)
    ax.set_xlim([lims[0], lims[1]])
    ax.set_ylim([lims[0], lims[1]])
    plt.title('Welfare Ranges ({})'.format(title_str))
    plt.grid()
    plt.show()

    # Save figure if enabled
    if save_en:
        name = './Plots/%s_Heatmap.png' % title_str
        fig.savefig(name)


def compute_summary_stats_arr(samples_arr, print_en=False, name=""):
    """
    Compute summary stats for a numpy array.

        Parameters_
            samples_arr:     Array of input samples
            print_en:        Flag to enable printing results
            name:            Title to print with results

        Returns_
            sum_stats:       [5, 25, 50, 75, 95] percentiles

    """
    sum_stats = np.percentile(samples_arr, [5, 25, 50, 75, 95])
    if print_en:
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.3e}".format(x)})
        print("Summary Statistics: {}".format(name))
        print("5th, 25th, 50th, 75th, 95th percentiles:")
        print("{}".format(sum_stats))
        print("Mean:")
        print(np.mean(samples_arr))
    return sum_stats
