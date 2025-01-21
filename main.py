from p_vqvae.experiments import *
import matplotlib

# Define the location to store the dataset
dataset_path = "./mnist_data"
matplotlib.use('Agg')

if __name__ == "__main__":
    from sklearn import datasets

    # general attack evaluation
    #domias(challenger_seed=0, data_type_challenger="3D_MRI", data_type_adversary="2D_MRI", plot_roc=True)  # Fig 6
    #domias(challenger_seed=0, data_type_challenger="3D_MRI", data_type_adversary="3D_MRI", plot_roc=True)  # Fig 6

    # vary knowledge of adversary
    #domias_vary_knowledge(n_a_mode="n_c", challenger_seed=0, data_type_challenger="3D_MRI", data_type_adversary="2D_MRI", show_auc=True, plot_log_p_r=True) # Fig 7
    #domias_vary_knowledge(n_a_mode="all" ,challenger_seed=0, data_type_challenger="3D_MRI", data_type_adversary="2D_MRI", show_auc=True) # Fig 7

    # exagerate NSF overfitting to test hypothesis about overfitting of NSF
    # Try only increasing overfitting of raw NSF
    domias_exagerate_nsf_overfitting(show_auc=True, challenger_seed=0, data_type_challenger="3D_MRI", data_type_adversary="2D_MRI")

    # check outliers
    domias_outlier_3d_to_2d(n_target_seeds=100, adjust_n_targets=True, show_auc=True)
    #domias_outlier_3d_to_2d(n_target_seeds=2, adjust_n_targets=False, show_auc=True)  # fig 5?

    # show NSF samples, if they both look shitty increase epochs_nsf to 200 or 400
    show_nsf_samples(file_name="data/plots/NSF_raw_sample_3D.png", data_type_challenger="3D_MRI", data_type_adversary="3D_MRI")
    show_nsf_samples(file_name="data/plots/NSF_raw_sample_2D.png", data_type_challenger="3D_MRI", data_type_adversary="2D_MRI")
