
from p_vqvae.experiments import *
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms

# Define the location to store the dataset
dataset_path = "./mnist_data"

import matplotlib
matplotlib.use('Agg')

if __name__ == "__main__":
    #show_synthetic_images()
    # show_nsf_samples()
    #plot_loss_distributions_domias(combine=True)
    #plot_diffs()
    #logan(1.0, plot_roc=True)
    #domias(1.0, plot_roc=True)
    #domias_outlier(1.0, plot_roc=True)
    #domias_vary_knowledge()
    #zdomias_vs_domias()
    #show_raw_image()
    #train_and_sample(2)  # this experiment: actnorm activated
    #show_synthetic_images()
    #multiple_challenger_seeds(n=4, adversary_knowledge=1.0, show_roc=True, show_mia_score_hist=True)  # this experiment: actnorm activated
    #domias_multi_seed_outlier(1.0)  # is the
    #domias_vary_knowledge()
    #plot_learning_curves_nsf()
    #plot_learning_curve_vqvae_and_transformer()
    #plot_learning_curve_vqvae_and_transformer()
    #show_synthetic_images()
    # mnist_train = datasets.MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    #domias_multiseed(n=4, show_roc=True, show_mia_score_hist=True)
    #domias_overfitting()

    domias_digits(plot_roc=True)


