import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import auc
from p_vqvae.mia import Challenger, AdversaryDOMIAS, AdversaryZCalibratedDOMIAS, AdversaryZCalibratedDOMIAS2, \
    AdversaryLiRANSF, AdversaryLiRAClassifier, AdversaryLOGAN, MultiSeedAdversaryDOMIAS
from p_vqvae.visualise import show_roc_curve, plot_generated_images, plot_reconstructions, show_roc_curve_std, \
                               show_auc_tpr_plot
from p_vqvae.networks import train_transformer_and_vqvae
from p_vqvae.dataloader import get_train_loader, get_train_val_loader, Atlas3dDataSet
from p_vqvae.neural_spline_flow import OptimizedNSF
from p_vqvae.utils import select_tpr_at_low_fprs

# don't forget to clear model outputs folder before implementing
# also check kwargs before implementation
TEST_MODE = True  # turn off for real attack

include_targets_in_reference_dataset = None  # either True, False or None (=included at random)

device = "cuda:4"
mpl.rcParams['axes.labelsize'] = 17
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
n_atlas = 955
n_digits = 1796
low_fpr = 0.001
now = datetime.now()
results_path = "data/results"

challenger_kwargs_mri = {
    "n_c": 200, #n_atlas // 2,  # number of raw images in dataloader of challenger
    "m_c": 1000, #n_atlas // 2,  # number of synthetic images
    "n_targets": n_atlas,
}
challenger_kwargs_digits = {
    "n_c": 100, # n_digits // 2,
    "m_c": 1000,
    "n_targets": n_digits,
}
adversary_kwargs_mri = {
    "background_knowledge": 1.0,
    "outlier_percentile": None,
#    "n_a": n_atlas,
}
raw_data_kwargs = {
    "root": "data/ATLAS_2",
    "cache_path": 'data/cache',
    "downsample": 4,  #downsample = 1
    "normalize": 1,
    "crop": ((8, 9), (12, 13), (0, 9)),
    "padding": ((1, 2), (0, 0), (1, 2))  # padding when no downsampling is done: = ((2, 2), (0, 0), (2, 2))
}
vqvae_train_loader_kwargs = {
    "batch_size": 4,  #
    "augment_flag": False,
    "num_workers": 1
}
nsf_train_loader_kwargs = {
    "batch_size": 4,  #
    "augment_flag": False,
    "num_workers": 1
}
shadow_model_train_loader_kwargs = {
    "batch_size": 2,
    "augment_flag": False,
    "num_workers": 1
}
vqvae_kwargs = {
    "n_epochs": 200,
    "early_stopping_patience": float('inf'),  # stop training after ... training steps of not improving val loss
    "val_interval": 2,
    "lr": 5e-5,  #
    "device": device,
#    "device_ids": [4,5],
    "dtype": torch.float32,
    "use_checkpointing": True,
    "commitment_cost": 0.25,
    "num_embeddings": 512,  #
    "embedding_dim": 64,  #
    "channels": (384, 512),  #
    "num_res_layers": 2,
    "num_res_channels": (384, 512),  #
    "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
    "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    "model_path": "model_outputs/mia"
}
transformer_kwargs = {
    "n_epochs": 200,
    "early_stopping_patience": float('inf'),
    "device": device,
#    "vqvae_device": "cuda:6",
#    "device_ids": [4,5],
    "lr": 5e-4,  #
    "attn_layers_dim": 96,  #
    "attn_layers_depth": 12,  #
    "attn_layers_heads": 12,  #
    "model_path": "model_outputs/mia"
}
nsf_kwargs_mri = {
    "model_path": "model_outputs/mia",
    "device": device,
    "eval_interval": 2,  # after how many steps evaluate on validation set, before: 20
    "early_stopping": 5, #float('inf'),  # after how many evaluation steps stop the training
    "steps_per_level": 10,
    "levels": 2,  # increase for non-downsampled dataset
    "multi_scale": True,
    "actnorm": True,
    "epochs": 400, #400,
    #   "batch_size": 16, #
    "learning_rate": 4e-4,
    "cosine_annealing": False,
    "eta_min": 0.,
    #   "num_steps": 1000,  # change in implementation
    "mask_type": "alternating",
    "one_by_one_conv": True,
    "coupling_layer_type": 'rational_quadratic_spline',
    "hidden_channels": 64,
    "use_resnet": False,
    "num_res_blocks": 5,  # If using resnet
    "resnet_batchnorm": True,
    "dropout_prob": 0.,
    "spline_parameters": {
        'num_bins': 4,  #
        'tail_bound': 1.,
        'min_bin_width': 1e-3,
        'min_bin_height': 1e-3,
        'min_derivative': 1e-3,
        'apply_unconditional_transform': False
    }
}

adversary_kwargs_digits = adversary_kwargs_mri.copy()
# adversary_kwargs_digits["n_a"] = n_digits

nsf_kwargs_digits = nsf_kwargs_mri.copy()
nsf_kwargs_digits["epochs"] = 200
nsf_kwargs_digits["cosine_annealing"] = False
nsf_kwargs_digits["learning_rate"] = 1.5e-4

#vqvae_kwargs_mri2d = vqvae_kwargs.copy()
#vqvae_kwargs_mri2d["epochs"] = 200

nsf_train_loader_kwargs_digits = nsf_train_loader_kwargs.copy()
nsf_train_loader_kwargs_digits["batch_size"] = 4

membership_classifier_kwargs = {
    "model_path": "model_outputs/mia"
}


challenger_standard_mri = None #Challenger(
#    **challenger_kwargs_mri,
#    raw_data_kwargs=raw_data_kwargs,
#    vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
#    vqvae_kwargs=vqvae_kwargs,
#    transformer_kwargs=transformer_kwargs,
#    seed=0,
#)


def log(name, content: dict, path="data/logs"):
    file = os.path.join(path, name + now.strftime("%d%m%Y_%H%M%S") + ".csv")
    df = pd.DataFrame.from_dict({"date": now.strftime("%d.%m.%Y_%H:%M:%S")})
    df.append(content)
    df.to_csv(file, mode='a', header=not os.path.exists(file))

def show_raw_mri_image():
    dataset = Atlas3dDataSet(**raw_data_kwargs)
    train_loader = get_train_loader(dataset, 2, False, 1)
    img = next(iter(train_loader))['image'].cpu().float()
    plot_generated_images(img, 1, "data/plots/raw_image.png")

def show_synthetic_images():
    """Plot sampled images of the VQVAE + transformer as sanity check."""
    t_vqvae = challenger_standard_mri.target_model
    syn = t_vqvae.create_synthetic_images(2)
    plot_generated_images(syn, 1, "data/plots/syn_image.png")

def show_nsf_samples():
    _adversary = AdversaryDOMIAS(challenger_standard_mri, raw_data_kwargs, nsf_train_loader_kwargs, background_knowledge=1.0,
                                 nsf_kwargs=nsf_kwargs_mri)
    raw_samples, syn_samples = _adversary.sample_nsf(2)
    plot_generated_images(raw_samples, 2, "data/plots/NSF_raw_samples.png")
    plot_generated_images(syn_samples, 2, "data/plots/NSF_syn_samples.png")

def plot_learning_curve_vqvae_and_transformer():
    seed = 10
    dataset = Atlas3dDataSet(**raw_data_kwargs)
    train_loader, val_loader = get_train_val_loader(dataset, **vqvae_train_loader_kwargs)

    vqvae_kwargs["val_interval"] = 1
    vqvae_kwargs["n_epochs"] = 100
    vqvae_kwargs["early_stopping_patience"] = float('inf')
    transformer_kwargs["val_interval"] = 1
    transformer_kwargs["n_epochs"] = 100
    transformer_kwargs["early_stopping_patience"] = float('inf')

    t_vqvae, vqvae = train_transformer_and_vqvae(train_loader, vqvae_kwargs, transformer_kwargs,
                                                 saving_kwargs= {"seed": f"{seed}"}, val_loader=val_loader, seed=seed,
                                                 return_vqvae=True)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 6)
    ax[0].plot(vqvae.epoch_recon_loss_list, color="blue", label="Training loss")
    ax[0].plot(vqvae.val_recon_epoch_loss_list, color="red", label="Validation loss")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend(loc="upper right")
    ax[0].title.set_text(f"First Stage")

    ax[1].plot(t_vqvae.epoch_ce_loss_list, color="blue", label="Training loss")
    ax[1].plot(t_vqvae.val_ce_epoch_loss_list, color="red", label="Validation loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend(loc="upper right")
    ax[1].title.set_text(f"Second Stage")
    print("saving figure")

    if vqvae.epoch_recon_loss_list is not None and t_vqvae.epoch_ce_loss_list is not None:
        plt.savefig("data/plots/learning_curves_vqvae.png")
        plt.clf()

def plot_learning_curves_nsf(data_type="raw"):
    if data_type == "raw":
        # seed = 69
        dataset = Atlas3dDataSet(**raw_data_kwargs)
        train_loader, val_loader = get_train_val_loader(dataset, **vqvae_train_loader_kwargs)
    elif data_type == "syn":
        raise NotImplementedError
    else:
        raise NotImplementedError

    nsf = OptimizedNSF(train_loader, val_loader, nsf_kwargs_mri)
    print("training nsf")
    nsf.train()

    train_log_density = nsf.log_density_list
    val_log_density = nsf.val_log_density_list
    train_loss = nsf.loss_list
    val_loss = nsf.val_loss_list

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(6, 6)
    ax.plot(train_log_density, color="blue", label="Training log density")
    ax.plot(val_log_density, color="red", label="Validation log density")
    ax.set_ylabel("log p")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")
    ax.title.set_text(f"NSF log densities")
    plt.tight_layout()
    plt.savefig(f"data/plots/learning_curves_NSF_{data_type}.png")
    plt.clf()


def plot_loss_distributions_domias(combine=True):
    _adversary = AdversaryDOMIAS(challenger_standard_mri, raw_data_kwargs, nsf_train_loader_kwargs, background_knowledge=1.0,
                                 nsf_kwargs=nsf_kwargs_mri)
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
    plt.tight_layout()
    plt.savefig("data/plots/loss_distribution_DOMIAS.png")

    if combine:
        plt.figure(figsize=(12, 8))
        plt.hist(log_p_s_members, color='blue', label='syn loss members', alpha=0.4, bins=85)
        plt.hist(log_p_s_nonmembers, color='tab:cyan', label='syn loss non-members', alpha=0.4, bins=85)
        plt.hist(log_p_r_members, color='tab:orange', label='raw loss members', alpha=0.4, bins=85)
        plt.hist(log_p_r_nonmembers, color='tab:red', label='raw loss non-members', alpha=0.4, bins=85)
        plt.legend()
        plt.xlabel("log p")
        plt.tight_layout()
        plt.savefig("data/plots/loss_distribution_DOMIAS_combined.png")


def plot_diffs(_adversary=None, title: str = None):
    if _adversary is None:
        _adversary = AdversaryDOMIAS(challenger_standard_mri, raw_data_kwargs, nsf_train_loader_kwargs,
                                     background_knowledge=1.0, nsf_kwargs=nsf_kwargs_mri)
    diffs_members = []
    diffs_non_members = []
    diffs = np.array(_adversary.diffs)
    diffs = (diffs - diffs.min())/(diffs.max() - diffs.min())
    for it, b in enumerate(_adversary.true_memberships):
        if b == 1:
            diffs_members.append(diffs[it])
        else:
            diffs_non_members.append(diffs[it])

    plt.figure(figsize=(8, 5))
    plt.hist(diffs_members, color='blue', label='Members', alpha=0.5, bins=75)
    plt.hist(diffs_non_members, color='red', label='Non-members', alpha=0.5, bins=75)
    plt.legend()
    plt.xlabel("log p_S - log p_R")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig("data/plots/loss_differences_DOMIAS.png")
    plt.clf()

def domias(data_type_challenger="3D_MRI", data_type_adversary="3D_MRI", challenger_seed=0, adversary_knowledge = 1.0,
           outlier_percentile=None, n_targets=None, plot_roc=False):
    if "digits" in data_type_challenger:
        _challenger_kwargs = challenger_kwargs_digits.copy()
        _adversary_kwargs = adversary_kwargs_digits.copy()
        _nsf_train_loader_kwargs = nsf_train_loader_kwargs_digits.copy()
        _nsf_kwargs = nsf_kwargs_digits.copy()
    else:
        _challenger_kwargs = challenger_kwargs_mri.copy()
        _adversary_kwargs = adversary_kwargs_mri.copy()
        _nsf_train_loader_kwargs = nsf_train_loader_kwargs.copy()
        _nsf_kwargs = nsf_kwargs_mri.copy()
    _adversary_kwargs["background_knowledge"] = adversary_knowledge
    _adversary_kwargs["outlier_percentile"] = outlier_percentile
    _challenger_kwargs["n_targets"] = n_targets if n_targets is not None else _challenger_kwargs["n_targets"]

    challenger = Challenger(
        **_challenger_kwargs,
        raw_data_kwargs=raw_data_kwargs,
        vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
        vqvae_kwargs=vqvae_kwargs,
        transformer_kwargs=transformer_kwargs,
        data_type=data_type_challenger,
        challenger_seed=challenger_seed,
    )

    _adversary = AdversaryDOMIAS(challenger, raw_data_kwargs, _nsf_train_loader_kwargs, nsf_kwargs=_nsf_kwargs,
                                 data_type=data_type_adversary, **_adversary_kwargs)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        data_type = data_type_challenger if data_type_challenger == data_type_adversary else data_type_challenger.replace("_MRI", "_") + data_type_adversary
        file_path = (f"data/plots/ROC_DOMIAS_{data_type}_knowledge{_adversary_kwargs['background_knowledge']}"
                     f"_outlier{_adversary_kwargs['outlier_percentile']}")
        title = f"ROC curve for {data_type}"
        show_roc_curve(tprs, fprs, f"DOMIAS_{data_type}", file_path=file_path + ".png", title=title)
        show_roc_curve(tprs, fprs, f"DOMIAS_{data_type}", low_fprs=True, file_path=file_path + "lowfprs.png", title=title)

    return {"tprs": tprs, "fprs": fprs, "auc": auc(fprs, tprs),
            "tpr_at_low_fpr": select_tpr_at_low_fprs(tprs, fprs, low_fpr), "low_fpr": low_fpr}


def domias_digits(adversary_knowledge=1.0, plot_roc=True, outlier_percentile=None, seed=0):
    challenger = Challenger(
        **challenger_kwargs_digits,
        raw_data_kwargs=raw_data_kwargs,
        vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
        vqvae_kwargs=vqvae_kwargs,
        transformer_kwargs=transformer_kwargs,
        data_type="digits",
        challenger_seed=seed,
    )
    _adversary = AdversaryDOMIAS(challenger, raw_data_kwargs, nsf_train_loader_kwargs_digits, adversary_knowledge,
                                 outlier_percentile, nsf_kwargs=nsf_kwargs_digits)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"DOMIAS Digits")
        show_roc_curve(tprs, fprs, f"DOMIAS Digits", low_fprs=True)
    return tprs, fprs, _adversary.log_p_s, _adversary.log_p_r, _adversary.true_memberships


def digits_show_synthetic_images():
    challenger = Challenger(
        **challenger_kwargs_digits,
        raw_data_kwargs=raw_data_kwargs,
        vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
        vqvae_kwargs=vqvae_kwargs,
        transformer_kwargs=transformer_kwargs,
        data_type="digits",
        challenger_seed=0,
    )
    """Plot sampled images of the VQVAE + transformer as sanity check."""
    t_vqvae = challenger.target_model
    syn = t_vqvae.create_synthetic_images(2)
    plot_generated_images(syn, 1, "data/plots/syn_image_digits1.png", dim=2)


def domias_multiseed(n_seeds=4, _challenger_kwargs: dict = None, _adversary_kwargs: dict = None, _vqvae_kwargs: dict =None,
                     _transformer_kwargs: dict = None, show_roc=False, show_mia_score_hist=False, data_type="MRI"):
    """Samples multiple victim datasets using n_seeds number of seeds."""
    if _challenger_kwargs is None:
        _challenger_kwargs = challenger_kwargs_mri if "MRI" in data_type else challenger_kwargs_digits
    if _adversary_kwargs is None:
        _adversary_kwargs = adversary_kwargs_mri if "MRI" in data_type else adversary_kwargs_digits
    if _vqvae_kwargs is None:
        _vqvae_kwargs = vqvae_kwargs
    if _transformer_kwargs is None:
        _transformer_kwargs = transformer_kwargs

    log_p_s_list, log_p_r_list = [], []
    tprs_all, fprs_all = [], []
    auc_list, tpr_at_low_fpr = [], []
    true_memberships_list = []
    _challenger = None
    for seed in range(n_seeds):
        if data_type == "MRI":
            _challenger = Challenger(
                **_challenger_kwargs,
                raw_data_kwargs=raw_data_kwargs,
                vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
                vqvae_kwargs=_vqvae_kwargs,
                transformer_kwargs=_transformer_kwargs,
                challenger_seed=seed,
            )
            _adversary = AdversaryDOMIAS(_challenger, raw_data_kwargs, nsf_train_loader_kwargs, **_adversary_kwargs,
                                         nsf_kwargs = nsf_kwargs_mri)
        elif data_type == "digits":
            _challenger = Challenger(
                **_challenger_kwargs,
                raw_data_kwargs=raw_data_kwargs,
                vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
                vqvae_kwargs=_vqvae_kwargs,
                transformer_kwargs=_transformer_kwargs,
                data_type="digits",
                challenger_seed=seed,
            )
            _adversary = AdversaryDOMIAS(_challenger, raw_data_kwargs, nsf_train_loader_kwargs,
                                               **_adversary_kwargs, nsf_kwargs = nsf_kwargs_digits)
        else:
            raise NotImplementedError(f"data_type {data_type} not implemented.")
        tprs, fprs = _adversary.tprs, _adversary.fprs

        tprs_all.append(tprs)
        fprs_all.append(fprs)
        auc_list.append(_adversary.auc)
        log_p_s_list.extend(list(_adversary.log_p_s))
        log_p_r_list.extend(list(_adversary.log_p_r))
        true_memberships_list.extend(_adversary.true_memberships)

        tp = select_tpr_at_low_fprs(tprs, fprs, low_fpr)
        tpr_at_low_fpr.append(tp)

    log_p_s, log_p_r = np.array(log_p_s_list), np.array(log_p_r_list)
    adversary = MultiSeedAdversaryDOMIAS(log_p_s,
                                         log_p_r,
                                         true_memberships_list,
                                         _challenger,
                                         raw_data_kwargs,
                                         nsf_train_loader_kwargs,
                                         **_adversary_kwargs,
                                         )
    #tprs_mean, fprs_mean, = adversary.tprs, adversary.fprs
    tpr_at_lowfpr_mean = np.mean(np.array(tpr_at_low_fpr))
    tpr_at_lowfpr_std = np.std(np.array(tpr_at_low_fpr))
    tprs_mean = np.mean(np.array(tprs_all), axis=0)
    fprs_mean = np.mean(np.array(fprs_all), axis=0)
    tprs_std = np.std(np.array(tprs_all), axis=0)
    auc_mean = adversary.auc
    auc_std = np.std(np.array(auc_list), axis=0)

    if show_roc:
        show_roc_curve_std(tprs=tprs_mean, fprs=fprs_mean, std1=tprs_std, label=f"DOMIAS {data_type}", low_fprs=False)
        show_roc_curve_std(tprs=tprs_mean, fprs=fprs_mean, std1=tprs_std, label=f"DOMIAS {data_type}", low_fprs=True)
    if show_mia_score_hist:
        plot_diffs(adversary, title=f"{data_type} data")

    return {"tprs": tprs_all, "fprs": fprs_all, "tprs_mean": tprs_mean, "fprs_mean": fprs_mean, "tprs_std": tprs_std,
            "auc_mean": auc_mean, "auc_std": auc_std, "tpr_at_low_fpr_mean": tpr_at_lowfpr_mean,
            "tpr_at_low_fpr_std": tpr_at_lowfpr_std}


def domias_multi_target_seeds(n_target_seeds=10, **kwargs):
    """Keeps training and data seed constant. Varies the choice of targets."""
    df = pd.DataFrame(columns=['auc', 'tpr_at_low_fpr', 'low_fpr'])
    for target_seed in range(n_target_seeds):
        results = domias(**kwargs)
        df = pd.concat([df, pd.DataFrame(results)])
    return df


def domias_outlier_3d_to_2d(n_target_seeds=10, show_auc=True):
    percentiles = [1.0, .25, .1, .02]
    df = pd.DataFrame(columns=['mean_auc', 'std_auc', 'mean_tpr_at_low_fpr', 'std_tpr_at_low_fpr', 'low_fpr'])

    for percentile in percentiles:
        n_targets = int(np.ceil(percentiles[-1] * n_atlas / percentile))
        percentile = None if percentile == 1.0 else percentile
        df_res = domias_multi_target_seeds(
            n_target_seeds=n_target_seeds,
            data_type_challenger="3D_MRI",
            data_type_adversary="2D_MRI",
            challenger_seed=0,
            adversary_knowledge=1.0,
            outlier_percentile=percentile,
            n_targets=n_targets,
        )
        df_summary = pd.DataFrame({
            'mean_auc': df_res['auc'].mean(),
            'std_auc': df_res['auc'].std(),
            'mean_tpr_at_low_fpr': df_res['tpr_at_low_fpr'].mean(),
            'std_tpr_at_low_fpr': df_res['tpr_at_low_fpr'].std(),
            'low_fpr': df_res['low_fpr'].mean()
        }, index=[0])
        df = pd.concat([df, df_summary])

    latex_table = df.to_latex(index=False, float_format="{:.2f}".format)
    with open(os.path.join(results_path + f"outlier_3d_to_2d_{now}.txt", "w")) as file:
        file.write(latex_table)

    if show_auc:
        show_auc_tpr_plot(percentiles, df['mean_auc'], df['std_auc'], xlabel="Outlier percentile", ylabel="AUC")
        show_auc_tpr_plot(percentiles, df['mean_tpr_at_low_fpr'], df['std_tpr_at_low_fpr'], xlabel="Percentile", ylabel=f"TPR at {low_fpr} FPR")


def domias_outlier(adversary_knowledge=1.0, plot_roc=False, challenger=challenger_standard_mri):
    _adversary_all = AdversaryDOMIAS(challenger, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge, None,
                                     nsf_kwargs=nsf_kwargs_mri)
    tprs_all, fprs_all = _adversary_all.tprs, _adversary_all.fprs

    _adversary_25 = AdversaryDOMIAS(challenger, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge, .25,
                                    nsf_kwargs=nsf_kwargs_mri)
    tprs_25, fprs_25 = _adversary_25.tprs, _adversary_25.fprs

    _adversary_10 = AdversaryDOMIAS(challenger, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge, .1,
                                    nsf_kwargs=nsf_kwargs_mri)
    tprs_10, fprs_10 = _adversary_10.tprs, _adversary_10.fprs

    if plot_roc:
        show_roc_curve(tprs_all, fprs_all, f"All targets", tprs_25, fprs_25, f"25th percentile outliers", tprs_10,
                       fprs_10, f"10th percentile outliers")
        show_roc_curve(tprs_all, fprs_all, f"All targets", tprs_25, fprs_25, f"25th percentile outliers", tprs_10,
                       fprs_10, f"10th percentile outliers", low_fprs=True)
    return tprs_10, fprs_10


def domias_overfitting(n_seeds=4, show_roc=True, show_aucs=True):
    epoch_list = [(None, None), (100, 100), (200, 200)]
    mean_auc, std_auc = [], []
    mean_tpr_lf, std_tpr_lf = [], []
    res = []
    for epochs in epoch_list:
        if epochs[0] is None:
            vqvae_kwargs["n_epochs"] = 200
            vqvae_kwargs["early_stopping_patience"] = 5
        else:
            vqvae_kwargs["n_epochs"] = epochs[0]
            vqvae_kwargs["early_stopping_patience"] = float('inf')

        if epochs[1] is None:
            transformer_kwargs["n_epochs"] = 200
            transformer_kwargs["early_stopping_patience"] = 5
        else:
            transformer_kwargs["n_epochs"] = epochs[1]
            transformer_kwargs["early_stopping_patience"] = float('inf')

        adv_res = domias_multiseed(n_seeds,
                                   challenger_kwargs_mri,
                                   adversary_kwargs_mri,
                                   vqvae_kwargs,
                                   transformer_kwargs,
                                   show_roc=False,
                                   show_mia_score_hist=False)

        log_content = {**adv_res, **challenger_kwargs_mri, **adversary_kwargs_mri, **vqvae_kwargs,
                       **transformer_kwargs, **vqvae_train_loader_kwargs, **nsf_train_loader_kwargs}
        log("overfitting", log_content)
        res.append(adv_res)
        mean_auc.append(adv_res["auc_mean"])
        std_auc.append(adv_res["auc_std"])
        mean_tpr_lf.append(adv_res["tpr_at_lowfpr_mean"])
        std_tpr_lf.append(adv_res["tpr_at_lowfpr_std"])

    #if show_roc:
    #    show_roc_curve_std(res[0]["tprs_mean"], res[0]["fprs_mean"], f"early stopping", tprs_list[1], fprs_list[1], f"100 epochs",
    #                   tprs_list[2], fprs_list[2], f"200 epochs")
    #    show_roc_curve_std(tprs_list[0], fprs_list[0], f"early stopping", tprs_list[1], fprs_list[1], f"100 epochs",
    #                   tprs_list[2], fprs_list[2], f"200 epochs", low_fprs=True)
    if show_aucs:
        show_auc_tpr_plot([65, 100, 200], mean_auc, std_auc, xlabel="epoch", ylabel="AUC")
        show_auc_tpr_plot([65, 100, 200], mean_tpr_lf, std_tpr_lf, xlabel="epoch", ylabel=f"TPR at {low_fpr} FPR")


def domias_downsampling_2d():
    ds_factors = [1, 2, 3, 4]
    aucs, tprs_at_low_fprs = [], []

    for factor in ds_factors:

        adversary_kwargs_mri["downsample_2d"] = factor
        adversary_kwargs_mri["slice_type"] = "axial"

        challenger = Challenger(
            **challenger_kwargs_mri,
            raw_data_kwargs=raw_data_kwargs,
            vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
            vqvae_kwargs=vqvae_kwargs,
            transformer_kwargs=transformer_kwargs,
            data_type="3D_MRI",
            challenger_seed=0,
        )

        _adversary = AdversaryDOMIAS(challenger, raw_data_kwargs, nsf_train_loader_kwargs, nsf_kwargs=nsf_kwargs_mri,
                                     **adversary_kwargs_mri)
        tprs, fprs = _adversary.tprs, _adversary.fprs
        aucs.append(auc(fprs, tprs))
        tprs_at_low_fprs.append(select_tpr_at_low_fprs(tprs, fprs, low_fpr))

    show_auc_tpr_plot(ds_factors, aucs, xlabel="Down-sampling", ylabel="AUC")
    show_auc_tpr_plot(ds_factors, tprs_at_low_fprs, xlabel="Down-sampling", ylabel=f"TPR at {low_fpr} FPR")


def domias_training_set_size_MRI():
    n_members_list = [n_atlas // 8, n_atlas // 4, n_atlas // 2]

    for _n_members in n_members_list:
        challenger = Challenger(
            _n_members,
            n_atlas // 2,
            raw_data_kwargs,
            vqvae_train_loader_kwargs,
            vqvae_kwargs,
            transformer_kwargs,
            )


def logan(adversary_knowledge=1.0, plot_roc=False):
    _adversary = AdversaryLOGAN(challenger_standard_mri, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge,
                                nsf_kwargs=nsf_kwargs_mri)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"LOGAN, knowledge: {adversary_knowledge}")
        show_roc_curve(tprs, fprs, f"LOGAN, knowledge: {adversary_knowledge}", low_fprs=True)
    return tprs, fprs


def domias_with_augmentation(adversary_knowledge=0.0, plot_roc=False):
    _nsf_train_loader_kwargs = {
        "batch_size": 4,
        "augment_flag": True,
        "num_workers": 1
    }
    _adversary = AdversaryDOMIAS(challenger_standard_mri, raw_data_kwargs, _nsf_train_loader_kwargs, adversary_knowledge,
                                 nsf_kwargs=nsf_kwargs_mri)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def zdomias_p_r_z(adversary_knowledge=1.0, n_z=100, plot_roc=False):
    _adversary = AdversaryZCalibratedDOMIAS(challenger_standard_mri, raw_data_kwargs, nsf_train_loader_kwargs,
                                            adversary_knowledge, n_z, nsf_kwargs=nsf_kwargs_mri)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def zdomias_p_s_z(adversary_knowledge=1.0, n_z=100, plot_roc=False):
    _adversary = AdversaryZCalibratedDOMIAS2(challenger_standard_mri, raw_data_kwargs, nsf_train_loader_kwargs,
                                             adversary_knowledge, n_z, nsf_kwargs=nsf_kwargs_mri)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def domias_vary_knowledge():
    pass
    #tprs_1, fprs_1 = multiple_challenger_seeds(n=4, adversary_knowledge=1.0)
    #tprs_2, fprs_2 = multiple_challenger_seeds(n=4, adversary_knowledge=0.5)
    #tprs_3, fprs_3 = multiple_challenger_seeds(n=4, adversary_knowledge=0.0)
    #show_roc_curve(tprs_1, fprs_1, f"100% knowledge", tprs_2, fprs_2, f"50% knowledge", tprs_3,
    #               fprs_3, f"0% knowledge", )
    #show_roc_curve(tprs_1, fprs_1, f"100% knowledge", tprs_2, fprs_2, f"50% knowledge", tprs_3,
    #               fprs_3, f"0% knowledge", low_fprs=True)


#def zdomias_vs_domias():
#    tprs2, fprs2 = zdomias_p_r_z(1.0, n_z=50)
#    tprs1, fprs1, _, _, _ = domias(1.0)
#    show_roc_curve(tprs1, fprs1, f"DOMIAS", tprs2, fprs2, f"z-DOMIAS")
#    show_roc_curve(tprs1, fprs1, f"DOMIAS", tprs2, fprs2, f"z-DOMIAS", low_fprs=True)


#def zdomias_vary_z():
#    tprs1, fprs1 = zdomias_p_r_z(1.0, 20)
#    tprs2, fprs2 = zdomias_p_r_z(1.0, 50)
#    tprs3, fprs3 = zdomias_p_r_z(1.0, 100)
#    show_roc_curve(tprs1, fprs1, f"z-DOMIAS, n_Z=20", tprs2, fprs2, f"z-DOMIAS, n_Z=50", tprs3, fprs3,
#                   f"z-DOMIAS, n_Z=100")
#    show_roc_curve(tprs1, fprs1, f"z-DOMIAS, n_Z=20", tprs2, fprs2, f"z-DOMIAS, n_Z=50", tprs3, fprs3,
#                   f"z-DOMIAS, n_Z=100", low_fprs=True)


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


def LIRA_classifier(data_type="digits", adversary_knowledge=1.0, outlier_percentile=None, plot_roc=True):
    challenger = Challenger(
        **challenger_kwargs_digits,
        raw_data_kwargs=raw_data_kwargs,
        vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
        vqvae_kwargs=vqvae_kwargs,
        transformer_kwargs=transformer_kwargs,
        data_type=data_type,
        seed=0,
    )
    _adversary = AdversaryLiRAClassifier(challenger, offline=False, n_reference_models=2,
                                         background_knowledge=adversary_knowledge,
                                         shadow_model_train_loader_kwargs=vqvae_train_loader_kwargs,
                                         membership_classifier_kwargs=membership_classifier_kwargs,
                                         outlier_percentile=outlier_percentile)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"DOMIAS Digits")
        show_roc_curve(tprs, fprs, f"DOMIAS Digits", low_fprs=True)
    return tprs, fprs, _adversary.log_p_s, _adversary.log_p_r, _adversary.true_memberships

#def LIRA_NSF(plot_roc=True):
#    challenger = Challenger(
#        n_members,
#        m_syn_images,
#        raw_data_kwargs,
#        vqvae_train_loader_kwargs,
#        vqvae_kwargs,
#        transformer_kwargs,
#        n_targets=100
#    )
#    _adversary = AdversaryLiRANSF(
#       challenger,
#        offline=False,
#        n_reference_models=1,
#        background_knowledge=1.0,
#        shadow_model_train_loader_kwargs=shadow_model_train_loader_kwargs,
#    )
#
#    tprs, fprs = _adversary.tprs, _adversary.fprs
#    if plot_roc:
#        show_roc_curve(tprs, fprs, f"LiRA NSF adversary knowledge: {1.0}")
#
#    return tprs, fprs


if __name__ == "__main__":
    show_synthetic_images()
    plot_loss_distributions_domias(combine=True)
    plot_diffs()
    domias(1.0, plot_roc=True)
    domias_vary_knowledge()
    zdomias_vs_domias()
