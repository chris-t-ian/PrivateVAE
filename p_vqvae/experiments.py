import matplotlib.pyplot as plt
import torch
import pandas as pd
from p_vqvae.mia import Challenger, AdversaryDOMIAS, AdversaryZCalibratedDOMIAS, AdversaryZCalibratedDOMIAS2, \
    AdversaryLiRANSF, AdversaryLiRAClassifier
from p_vqvae.visualise import show_roc_curve, plot_generated_images, plot_reconstructions
from p_vqvae.networks import train_transformer_and_vqvae

# don't forget to clear model outputs folder before implementing
# also check kwargs before implementation
TEST_MODE = True  # turn off for real attack

include_targets_in_reference_dataset = None  # either True, False or None (=included at random)

if TEST_MODE:
    epochs_vqvae = 100
    epochs_transformer = 50
    downsample = 4
    m_syn_images = 955 // 2
    n_targets = 955
else:
    epochs_vqvae = 100
    epochs_transformer = 50
    downsample = 1
    m_syn_images = 955 // 2
    n_targets = 955

raw_data_kwargs = {
    "root": "/home/chrsch/P_VQVAE/data/ATLAS_2",
    "cache_path": '/home/chrsch/P_VQVAE/data/cache/',
    "downsample": downsample,
    "normalize": 1,
    "crop": ((8, 9), (12, 13), (0, 9)),
    "padding": ((1, 2), (0, 0), (1, 2))}
vqvae_train_loader_kwargs = {
    "batch_size": 1,
    "augment_flag": True,
    "num_workers": 2
}
nsf_train_loader_kwargs = {
    "batch_size": 1,
    "augment_flag": False,
    "num_workers": 1
}
shadow_model_train_loader_kwargs = {
    "batch_size": 8,
    "augment_flag": False,
    "num_workers": 1
}
training_vqvae_kwargs = {
    "n_epochs": epochs_vqvae,
    "lr": 1e-4,
    "multiple_devices": False,
    "dtype": torch.float32,
    "use_checkpointing": True,
    "commitment_cost": 0.05,
    "num_embeddings": 256,
    "embedding_dim": 32,
    "num_res_layers": 2,
    "num_res_channels": (256, 256),
    "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
    "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    "model_path": "/home/chrsch/P_VQVAE/model_outputs/lira"
}
training_transformer_kwargs = {
    "n_epochs": epochs_transformer,
    "attn_layers_dim": 96,
    "attn_layers_depth": 12,
    "attn_layers_heads": 8,
    "lr": 5e-4,  # todo: fix "got multiple values for keyword argument 'lr'"
    "model_path": "/home/chrsch/P_VQVAE/model_outputs/lira"
}
membership_classifier_kwargs = {
    "model_path": "/home/chrsch/P_VQVAE/model_outputs/lira",
}

# always use the same challenger:
challenger_half = Challenger(
        955 // 2,
        m_syn_images,
        raw_data_kwargs,
        vqvae_train_loader_kwargs,
        training_vqvae_kwargs,
        training_transformer_kwargs,
        n_targets
    )
#challenger_third = Challenger(
#        955 // 3,
#        m_syn_images,
#        raw_data_kwargs,
#        vqvae_train_loader_kwargs,
#        training_vqvae_kwargs,
#        training_transformer_kwargs,
#        n_targets
#    )


def show_synthetic_images():
    """Plot sampled images of the VQVAE + transformer as sanity check."""
    t_vqvae = challenger_half.t_vqvae
    syn = t_vqvae.create_synthetic_images(2)
    plot_generated_images(syn, n=2)


def plot_loss_distributions_domias(combine=True):
    _adversary = AdversaryDOMIAS(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs, background_knowledge=1.0)
    log_p_s_members = []
    log_p_r_members = []
    log_p_s_nonmembers = []
    log_p_r_nonmembers = []
    for it, b in enumerate(_adversary.true_memberships):
        if b == 1:
            log_p_s_members.append(_adversary.log_p_s[it])
            log_p_r_members.append(_adversary.log_p_r[it])
        else:
            log_p_s_nonmembers.append(_adversary.log_p_s[it])
            log_p_r_nonmembers.append(_adversary.log_p_r[it])

    if combine:
        plt.figure(figsize=(16, 9))
        plt.hist(log_p_s_members, color='blue', label='syn loss members', alpha=0.3, bins=85)
        plt.hist(log_p_s_nonmembers, color='tab:cyan', label='syn loss non-members', alpha=0.3, bins=85)
        plt.hist(log_p_r_members, color='tab:orange', label='raw loss members', alpha=0.3, bins=85)
        plt.hist(log_p_r_nonmembers, color='tab:red', label='raw loss non-members', alpha=0.3, bins=85)
        plt.legend()
        plt.xlabel("log p")
    else:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(16, 9)
        ax[0, 0].hist(log_p_s_members, color='blue', label='syn loss members', alpha=0.5, bins=50)
        ax[0, 0].hist(log_p_r_members, color='red', label='raw loss members', alpha=0.5, bins=50)
        ax[0, 0].legend(loc="upper left")

        ax[0, 1].hist(log_p_s_nonmembers, color='blue', label='syn loss non-members', alpha=0.5, bins=50)
        ax[0, 1].hist(log_p_r_nonmembers, color='red', label='raw loss non-members', alpha=0.5, bins=50)
        ax[0, 1].legend(loc="upper left")

        ax[1, 0].hist(log_p_s_members, color='blue', label='syn loss members', alpha=0.5, bins=50)
        ax[1, 0].hist(log_p_s_nonmembers, color='red', label='syn loss non members', alpha=0.5, bins=50)
        ax[1, 0].legend(loc="upper right")

        ax[1, 1].hist(log_p_r_members, color='blue', label='raw loss members', alpha=0.5, bins=50)
        ax[1, 1].hist(log_p_r_nonmembers, color='red', label='raw loss nonmembers', alpha=0.5, bins=50)
        ax[0, 1].legend(loc="upper right")
        plt.xlabel("log p")
    plt.show()


def domias(adversary_knowledge=1.0, plot_roc=False):
    _adversary = AdversaryDOMIAS(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"Augmentation and adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def domias_with_augmentation(adversary_knowledge=0.0, plot_roc=False):
    _nsf_train_loader_kwargs = {
        "batch_size": 1,
        "augment_flag": True,
        "num_workers": 1
    }
    _adversary = AdversaryDOMIAS(challenger_half, raw_data_kwargs, _nsf_train_loader_kwargs, adversary_knowledge)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def zdomias_p_r_z(adversary_knowledge=1.0, n_z=100, plot_roc=False):
    _adversary = AdversaryZCalibratedDOMIAS(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge,
                                            n_z)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def zdomias_p_s_z(adversary_knowledge=1.0, n_z=100, plot_roc=False):
    _adversary = AdversaryZCalibratedDOMIAS2(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge,
                                             n_z)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def domias_vary_knowledge():
    tprs_1, fprs_1 = domias(1.0)
    tprs_2, fprs_2 = domias(0.5)
    tprs_3, fprs_3 = domias(0.0)
    show_roc_curve(tprs_1, fprs_1, f"100% adversary knowledge", tprs_2, fprs_2, f"50% adversary knowledge", tprs_3,
                   fprs_3, f"0% adversary knowledge")


def zdomias_vs_domias():
    tprs2, fprs2 = zdomias_p_r_z(1.0)
    tprs1, fprs1 = domias(1.0)
    show_roc_curve(tprs1, fprs1, f"DOMIAS", tprs2, fprs2, f"z-DOMIAS")
    show_roc_curve(tprs1, fprs1, f"DOMIAS", tprs2, fprs2, f"z-DOMIAS", low_fprs=True)


def zdomias_vary_z():
    tprs1, fprs1 = zdomias_p_r_z(1.0, 20)
    tprs2, fprs2 = zdomias_p_r_z(1.0, 50)
    tprs3, fprs3 = zdomias_p_r_z(1.0, 100)
    show_roc_curve(tprs1, fprs1, f"z-DOMIAS, n_Z=20", tprs2, fprs2, f"z-DOMIAS, n_Z=50", tprs3, fprs3,
                   f"z-DOMIAS, n_Z=100")
    show_roc_curve(tprs1, fprs1, f"z-DOMIAS, n_Z=20", tprs2, fprs2, f"z-DOMIAS, n_Z=50", tprs3, fprs3,
                   f"z-DOMIAS, n_Z=100", low_fprs=True)


def zdomias_vary_knowledge():
    tprs_1, fprs_1 = zdomias_p_r_z(1.0)
    tprs_2, fprs_2 = zdomias_p_r_z(0.5)
    tprs_3, fprs_3 = zdomias_p_r_z(0.0)
    show_roc_curve(tprs_1, fprs_1, f"100% adversary knowledge", tprs_2, fprs_2, f"50% adversary knowledge", tprs_3,
                   fprs_3, f"0% adversary knowledge")
    show_roc_curve(tprs_1, fprs_1, f"100% adversary knowledge", tprs_2, fprs_2, f"50% adversary knowledge", tprs_3,
                   fprs_3, f"0% adversary knowledge", low_fprs=True)


def zdomias_repeat():
    tprs1, fprs1 = zdomias_p_r_z(1.0, 455)
    tprs2, fprs2 = zdomias_p_r_z(1.0, 455)
    tprs3, fprs3 = zdomias_p_r_z(1.0, 455)
    show_roc_curve(tprs1, fprs1, f"z-DOMIAS1", tprs2, fprs2, f"z-DOMIAS2", tprs3, fprs3, f"z-DOMIAS3", )


def zdomias_p_r_z_vs_p_s_z():
    tprs1, fprs1 = zdomias_p_r_z(1.0, 455)
    tprs2, fprs2 = zdomias_p_s_z(1.0, 455)
    tprs3, fprs3 = zdomias_p_s_z(1.0, 455)
    show_roc_curve(tprs1, fprs1, f"p_r_z-DOMIAS", tprs2, fprs2, f"p_s_z-DOMIAS1", tprs3, fprs3, f"p_s_z-DOMIAS2", )


def zdomias_p_s_z_repeat():
    tprs1, fprs1 = zdomias_p_s_z(1.0, 455)
    tprs2, fprs2 = zdomias_p_s_z(1.0, 455)
    tprs3, fprs3 = zdomias_p_s_z(1.0, 455)
    show_roc_curve(tprs1, fprs1, f"p_s_z-DOMIAS1", tprs2, fprs2, f"p_s_z-DOMIAS2", tprs3, fprs3, f"p_s_z-DOMIAS3", )


def LIRA_NSF(plot_roc=True):
    challenger = Challenger(
        955 // 2,
        m_syn_images,
        raw_data_kwargs,
        vqvae_train_loader_kwargs,
        training_vqvae_kwargs,
        training_transformer_kwargs,
        n_targets=100
    )
    _adversary = AdversaryLiRANSF(
        challenger,
        offline=False,
        n_reference_models=1,
        background_knowledge=1.0,
        shadow_model_train_loader_kwargs=shadow_model_train_loader_kwargs,
    )

    tprs, fprs = _adversary.tprs, _adversary.fprs
    if plot_roc:
        show_roc_curve(tprs, fprs, f"LiRA NSF adversary knowledge: {1.0}")

    return tprs, fprs


if __name__ == "__main__":
    # zdomias_vs_domias()
    # domias(1.0, plot_roc=True)
    # show_synthetic_images()
    # domias_vary_knowledge()
    # zdomias_repeat()
    # zdomias_vary_z()
    # plot_loss_distributions_domias(combine=True)
    # zdomias_p_r_z_vs_p_s_z()
    # zdomias_p_s_z_repeat()
    # domias_with_augmentation(plot_roc=True)
    LIRA_NSF()
