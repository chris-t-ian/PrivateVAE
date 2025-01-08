from p_vqvae.dataloader import RawDataSet, SyntheticDataSet, DigitsDataSet, get_train_loader, get_train_val_loader
from p_vqvae.networks import train_transformer_and_vqvae, MembershipClassifier
from p_vqvae.neural_spline_flow import OptimizedNSF
from p_vqvae.utils import subset_to_sha256_key, calculate_AUC
from scipy.stats import multivariate_normal
from torch.utils.data import ConcatDataset
from sklearn.metrics import roc_curve, auc
import numpy as np
import random
import torch
import os

test_mode = True
n_atlas = 955


class ChallengerRawDataSet(RawDataSet):
    def __init__(self, seed, n_train, **kwargs):
        """

        :param seed: seed for which to select training subjects
        :param n_train: number of images to train on
        :param kwargs: kwargs for training dataset
        """
        super().__init__(mode="full", **kwargs)
        self.n_train = n_train
        self.challenger_ids, self.non_challenger_ids = self.sample_challenger_dataset(seed)

    def __getitem__(self, idx):
        """Overwriting __get_item__ method, so that the idx refers to index within the challenger dataset. Maps input
        "idx" to items in the challenger dataset."""
        i = self.challenger_ids[idx]  # mapping of idx to item in challenger dataset
        image = np.copy(self.data[i])

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        """Return length of challenger dataset instead of the size of the whole data distribution."""
        return len(self.challenger_ids)

    def sample_challenger_dataset(self, seed):
        """Take entire (memory mapped) dataset and sample n_train images. Return sampled dataset in random order and the
        rest of the dataset distribution without the challenger dataset."""
        random.seed(seed)  # fixed seed for sampling datasets
        random_challenger_ids = random.sample(range(0, n_atlas), self.n_train)
        non_challenger_ids = [i for i in range(0, n_atlas) if i not in random_challenger_ids]

        # return distribution[random_challenger_ids], distribution[non_challenger_ids]
        return random_challenger_ids, non_challenger_ids


class ChallengerDigitsDataset(DigitsDataSet):
    def __init__(self, seed, n_train, **kwargs):
        """

        :param seed: seed for which to select training subjects
        :param n_train: number of images to train on
        :param kwargs: kwargs for training dataset
        """
        super().__init__(**kwargs)
        self.n_train = n_train
        self.challenger_ids, self.non_challenger_ids = self.sample_challenger_dataset(seed)

    def __getitem__(self, idx):
        """Overwriting __get_item__ method, so that the idx refers to index within the challenger dataset. Maps input
        "idx" to items in the challenger dataset."""
        i = self.challenger_ids[idx]  # mapping of idx to item in challenger dataset
        image = np.copy(self.data[i])

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        """Return length of challenger dataset instead of the size of the whole data distribution."""
        return len(self.challenger_ids)

    def sample_challenger_dataset(self, seed):
        """Take entire (memory mapped) dataset and sample n_train images. Return sampled dataset in random order and the
        rest of the dataset distribution without the challenger dataset."""
        random.seed(seed)  # fixed seed for sampling datasets
        random_challenger_ids = random.sample(range(0, n_atlas), self.n_train)
        non_challenger_ids = [i for i in range(0, n_atlas) if i not in random_challenger_ids]

        # return distribution[random_challenger_ids], distribution[non_challenger_ids]
        return random_challenger_ids, non_challenger_ids


class Challenger:
    def __init__(
            self,
            n_c,  # number of raw images in dataloader of challenger
            m_c,  # number of synthetic images
            raw_data_kwargs: dict,
            vqvae_train_loader_kwargs: dict,
            vqvae_kwargs: dict,
            transformer_kwargs: dict,
            n_targets=n_atlas,
            seed=None,
    ):
        """
        :param n_c: number of raw images in dataloader of challenger
        :param m_c: number of synthetic images of challenger
        :param raw_data_kwargs: arguments passed to RawDataSet
        :param vqvae_train_loader_kwargs: arguments passed to VQVAE train loader
        :param vqvae_kwargs: arguments passed to VQVAE Trainer
        :param transformer_kwargs: arguments passed to Transformer Trainer
        :param n_targets: number of targets
        :param seed: seed for which to select training subjects
        """
        self.m_c = m_c
        self.data_seed = 420 if seed is None else seed
        self.training_seed = 69 if seed is None else seed

        self.raw_data_kwargs = raw_data_kwargs
        self.vqvae_train_loader_kwargs = vqvae_train_loader_kwargs
        self.training_vqvae_kwargs = vqvae_kwargs
        self.training_transformer_kwargs = transformer_kwargs

        # sample challenger dataset using fixed seed
        challenger_ds = self.get_challenger_raw_dataset(n_c, **raw_data_kwargs)
        challenger_train_loader = get_train_loader(challenger_ds, **vqvae_train_loader_kwargs)

        self.challenger_ids = challenger_ds.challenger_ids
        self.non_challenger_ids = challenger_ds.non_challenger_ids
        self.target_ids, self.target_memberships = self.sample_n_random_target_ids(n_targets)

        # validation set
        challenger_val_ds = self.get_challenger_raw_dataset_val(**raw_data_kwargs)
        challenger_val_loader = get_train_loader(challenger_val_ds, **vqvae_train_loader_kwargs)

        # train challenger model, train VQVAE and transformer decoder at once using a fixed seed
        self.t_vqvae = train_transformer_and_vqvae(challenger_train_loader,
                                                   val_loader=challenger_val_loader,
                                                   vqvae_kwargs=vqvae_kwargs,
                                                   transformer_kwargs=transformer_kwargs,
                                                   saving_kwargs={"ds": raw_data_kwargs["downsample"], "challenger": 1,
                                                                  "epochs": f"{vqvae_kwargs['n_epochs']}_"
                                                                    f"{transformer_kwargs['n_epochs']}",
                                                                  "stopping": f"{vqvae_kwargs['early_stopping_patience']}_"
                                                                  f"{transformer_kwargs['early_stopping_patience']}",
                                                                  "n": f"{n_c}", "seed": self.training_seed},
                                                   seed=self.training_seed,
                                                   )

        # load m synthetic datapoints or generate and save them
        path_to_syn_data = os.path.join(transformer_kwargs["model_path"], "synthetic_data")
        syn_data_file = os.path.join(path_to_syn_data, f"challenger_syn_seed{self.training_seed}_n{n_c}_"
                                                       f"e{vqvae_kwargs['n_epochs']}_{transformer_kwargs['n_epochs']}"
                                                       f"es{vqvae_kwargs['early_stopping_patience']}_"
                                                       f"{transformer_kwargs['early_stopping_patience']}.npy")

        if not os.path.isdir(path_to_syn_data):
            os.makedirs(path_to_syn_data)
        print("file: ", syn_data_file)
        if not os.path.isfile(syn_data_file):
            challenger_syn_imgs = self.t_vqvae.create_synthetic_images(self.m_c)
            np.save(syn_data_file, challenger_syn_imgs)

        self.challenger_syn_dataset = self.get_challenger_syn_dataset(syn_data_file)

        # self.n_c = challenger_ds.__len__()  # size of challenger dataset

    def sample_n_random_target_ids(self, _n):
        """Sample N targets at once. Flip random bit b. If if b = 1: choose target in challenger dataset. If b = 0:
         choose t not in dataset and remove target from non-challenger data to avoid unintended membership.
         Return indices of targets and their membership label."""
        if test_mode:
            random.seed()
        else:
            random.seed()  # fresh seed

        if _n >= n_atlas:  # take all ids as targets
            target_ids = [id for id in range(0, n_atlas)]
            random.shuffle(target_ids)
            target_memberships = [1 if id in self.challenger_ids else 0 for id in target_ids]
        else:
            print("Number of targets: ", _n)
            target_ids = []
            target_memberships = [random.getrandbits(1) for _ in range(_n)]

            challenger_ids = self.challenger_ids.copy()          # create a copy of ids so that it can be changed
            non_challenger_ids = self.non_challenger_ids.copy()  # without changing the class attribute self. _ids

            for b in target_memberships:
                if b == 1:
                    target_id = random.sample(challenger_ids, 1)[0]
                    challenger_ids.remove(target_id)  # remove from list to avoid drawing the same target again
                else:
                    target_id = random.sample(non_challenger_ids, 1)[0]
                    non_challenger_ids.remove(target_id)  # remove from list to avoid drawing the same target again

                target_ids.append(target_id)

        return target_ids, target_memberships

    def get_challenger_raw_dataset(self, n_c, **kwargs):
        return ChallengerRawDataSet(self.data_seed, n_c, **kwargs)

    def get_challenger_raw_dataset_val(self, **kwargs):
        # why is here int(0.2 * len(self.non_challenger_ids))]?
        # because it's a validation set
        return AdversaryRawDataSet(self.non_challenger_ids[:int(0.2 * len(self.non_challenger_ids))], **kwargs)

    def get_challenger_syn_dataset(self, syn_file):
        return SyntheticDataSet(syn_file)


class ChallengerDigits(Challenger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_challenger_raw_dataset(self, n_c, **kwargs):
        return ChallengerDigitsDataset(self.data_seed, n_c, **kwargs)

    def get_challenger_raw_dataset_val(self, **kwargs):
        # why is here int(0.2 * len(self.non_challenger_ids))]?
        # because it's the validation set and challenger_ids all need to be in the training set
        return AdversaryDigitsDataSet(self.non_challenger_ids[:int(0.2 * len(self.non_challenger_ids))], **kwargs)

    def get_challenger_syn_dataset(self, syn_file):
        return SyntheticDataSet(syn_file)


class AdversaryRawDataSet(RawDataSet):
    def __init__(self, adversary_ids, **kwargs):
        super().__init__(mode="full", **kwargs)
        self.adversary_ids = adversary_ids

    def __getitem__(self, idx):
        """Overwrite __get_item__ method, so that the idx refers to index within the challenger dataset. Maps input
        "idx" to items in the challenger dataset."""
        i = self.adversary_ids[idx]  # mapping of idx to item in challenger dataset
        image = np.copy(self.data[i])

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        """Return length of challenger dataset instead of the size of the whole data distribution."""
        return len(self.adversary_ids)

    def getitem_by_id(self, id):
        """Mapping of id to the whole dataset."""
        image = np.copy(self.data[id])
        return image


class AdversaryDigitsDataSet(DigitsDataSet):
    def __init__(self, adversary_ids, **kwargs):
        super().__init__(**kwargs)
        self.adversary_ids = adversary_ids

    def __getitem__(self, idx):
        """Overwrite __get_item__ method, so that the idx refers to index within the challenger dataset. Maps input
        "idx" to items in the challenger dataset."""
        i = self.adversary_ids[idx]  # mapping of idx to item in challenger dataset
        image = np.copy(self.data[i])

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        """Return length of challenger dataset instead of the size of the whole data distribution."""
        return len(self.adversary_ids)

    def getitem_by_id(self, id):
        """Mapping of id to the whole dataset."""
        image = np.copy(self.data[id])
        return image


class AdversaryDOMIAS:
    def __init__(
            self,
            _challenger: Challenger,
            data_kwargs: dict,
            nsf_train_loader_kwargs: dict,
            background_knowledge: float = 0.0,
            outlier_percentile: float = None,
    ):
        # self.synthetic_train_loader = get_train_loader(_challenger.challenger_syn_dataset, **nsf_train_loader_kwargs)
        self.challenger = _challenger
        self.synthetic_train_loader, self.synthetic_val_loader = get_train_val_loader(
            _challenger.challenger_syn_dataset, **nsf_train_loader_kwargs)
        self.background_knowledge = background_knowledge
        self.challenger_ids = _challenger.challenger_ids
        self.non_challenger_ids = _challenger.non_challenger_ids
        self.target_ids = _challenger.target_ids
        self.true_memberships = self.get_true_memberships()
        self.thresholds = None
        self.adversary_ids = self.sample_adversary_ids(n=len(self.challenger_ids))
        self.non_adversary_ids = [i for i in range(0, n_atlas) if i not in self.adversary_ids]
        assert not bool(set(self.adversary_ids) & set(self.non_adversary_ids)), \
            (f"adversary_ids and non_adversary_ids overlap with "
             f"{len(set(self.adversary_ids) & set(self.non_adversary_ids))} elements")

        self.auxiliary_ds = self.get_adversary_raw_dataset(**data_kwargs)
        self.auxiliary_ds_val = self.get_adversary_raw_dataset_val(**data_kwargs)

        # self.adversary_train_loader = get_train_loader(self.adversary_ds, **nsf_train_loader_kwargs)
        self.adversary_train_loader = get_train_loader(self.auxiliary_ds, **nsf_train_loader_kwargs)
        self.adversary_val_loader = get_train_loader(self.auxiliary_ds_val, **nsf_train_loader_kwargs)

        # train model on raw and on synthetic dataset and calculate log densities for each target
        self.log_p_s, self.log_p_r = self.calculate_log_densities()

        if outlier_percentile is not None:
            self.log_p_s, self.log_p_r, self.target_ids, self.true_memberships = self.select_outliers(outlier_percentile)

        self.diffs = self.log_p_s - self.log_p_r
        self.tprs, self.fprs = self.roc_curve()

        # diffs = self.f_logx_normalized(self.log_p_s, self.log_p_r)
        # self.fprs, self.tprs, _ = roc_curve(np.array(self.true_memberships), diffs)
        self.auc = auc(self.fprs, self.tprs)
        print("auc: ", calculate_AUC(self.tprs, self.fprs))
        print("auc (sklearn): ", auc(self.fprs, self.tprs))

    def sample_adversary_ids(self, n):
        if test_mode:
            random.seed(40)
        else:
            random.seed()  # fresh seed

        challenger_ids = self.challenger_ids.copy()
        non_challenger_ids = self.non_challenger_ids.copy()

        if self.background_knowledge == 1.0:
            adversary_ids = challenger_ids
        elif self.background_knowledge == 0.5:
            adversary_ids = random.sample(challenger_ids, n // 2)
            adversary_ids.extend(random.sample(non_challenger_ids, n // 2))
        elif self.background_knowledge == 0.0:
            adversary_ids = non_challenger_ids
        else:
            n_challenger_ids = int(n * self.background_knowledge)
            assert 1 <= n_challenger_ids < n, "Invalid value given for background knowledge."
            adversary_ids = random.sample(challenger_ids, n_challenger_ids)
            adversary_ids.extend(random.sample(non_challenger_ids, n - n_challenger_ids))

        random.shuffle(adversary_ids)
        return adversary_ids

    def calculate_log_densities(self):
        nsf_syn = OptimizedNSF(self.synthetic_train_loader, self.synthetic_val_loader)
        syn_saving_keys = {"adversary": "DOMIAS", "syn": "", "sha": f"{subset_to_sha256_key(self.challenger_ids)}"}

        if nsf_syn.model_exists(**syn_saving_keys):
            nsf_syn.load(**syn_saving_keys)
        else:
            nsf_syn.train()
            nsf_syn.save(**syn_saving_keys)

        # for every target x get p_S(x) from the nsf-density estimator
        log_p_s = []
        for target_id in self.target_ids:
            target = self.auxiliary_ds.getitem_by_id(target_id)
            target = np.expand_dims(target, axis=0)
            target = torch.from_numpy(target).to(device=nsf_syn.device)
            # print("log_p_s val:", nsf_syn.eval_log_density(target).item())
            log_p_s.append(nsf_syn.eval_log_density(target).item())

        if "cuda" in str(nsf_syn.device):
            torch.cuda.empty_cache()

        nsf_raw = OptimizedNSF(self.adversary_train_loader, self.adversary_val_loader)
        raw_saving_keys = {"adversary": "DOMIAS", "raw": "", "sha": f"{subset_to_sha256_key(self.adversary_ids)}"}
        if nsf_raw.model_exists(**raw_saving_keys):
            nsf_raw.load(**raw_saving_keys)
        else:
            nsf_raw.train()
            nsf_raw.save(**raw_saving_keys)

        log_p_r = []
        for target_id in self.target_ids:
            target = self.auxiliary_ds.getitem_by_id(target_id)
            target = np.expand_dims(target, axis=0)
            target = torch.from_numpy(target).to(device=nsf_raw.device)
            log_p_r.append(nsf_raw.eval_log_density(target).item())

        if "cuda" in str(nsf_raw.device):
            torch.cuda.empty_cache()

        return np.array(log_p_s, dtype=np.float64), np.array(log_p_r, dtype=np.float64)

    def sample_nsf(self, n):
        nsf_raw = OptimizedNSF(self.adversary_train_loader, self.adversary_val_loader)
        raw_saving_keys = {"adversary": "DOMIAS", "raw": "", "sha": f"{subset_to_sha256_key(self.adversary_ids)}"}
        if nsf_raw.model_exists(**raw_saving_keys):
            nsf_raw.load(**raw_saving_keys)
        else:
            nsf_raw.train()
            nsf_raw.save(**raw_saving_keys)

        nsf_syn = OptimizedNSF(self.synthetic_train_loader, self.synthetic_val_loader)
        syn_saving_keys = {"adversary": "DOMIAS", "syn": "", "sha": f"{subset_to_sha256_key(self.challenger_ids)}"}

        if nsf_syn.model_exists(**syn_saving_keys):
            nsf_syn.load(**syn_saving_keys)
        else:
            nsf_syn.train()
            nsf_syn.save(**syn_saving_keys)

        raw_samples = nsf_raw.sample(n)
        syn_samples = nsf_syn.sample(n)

        return raw_samples, syn_samples

    def infer_membership_logx(self, gamma):
        diffs = self.f_logx_normalized(self.log_p_s, self.log_p_r)
        infered_memberships = []
        for diff in diffs:
            if diff >= gamma:
                b = 1
            else:
                b = 0
            infered_memberships.append(b)
        return infered_memberships

    @staticmethod
    def f_logx_normalized(log_a, log_b):
        """f(log(a)/log(b)) = log(a) - log(b); f(x): [min_diff:max_diff] -> [0, 1]"""
        diffs = log_a - log_b

        # normalize diffs:
        return (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs))

    def roc_curve(self, true=None, score=None, _range=None):
        tprs = []
        fprs = []
        true_memberships = np.array(self.true_memberships) if true is None else np.array(true)

        if self.thresholds is None:
            thresholds = np.arange(0.0, 1.002, 0.002)
        else:
            thresholds = self.thresholds

        for gamma in thresholds:
            inferred_memberships = np.array((self.infer_membership_logx(gamma)))

            tp = np.sum((true_memberships == 1) & (inferred_memberships == 1))
            tn = np.sum((true_memberships == 0) & (inferred_memberships == 0))
            fp = np.sum((true_memberships == 0) & (inferred_memberships == 1))
            fn = np.sum((true_memberships == 1) & (inferred_memberships == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tprs.append(tpr)
            fprs.append(fpr)

        print("tprs: ", tprs)
        print("fprs: ", fprs)
        return tprs, fprs

    def select_outliers(self, percentile=0.2):
        """Select outliers of the dataset by selecting targets with low densities."""
        target_ids = np.array(self.target_ids)
        true_memberships = np.array(self.true_memberships)

        num_elements = int(np.ceil(percentile * len(target_ids)))

        # Get sorted indices of the densities array (ascending order)
        sorted_indices = np.argsort(self.log_p_r)

        # Get the indices for the top 20% (from the end of the sorted array)
        top_20_indices = sorted_indices[-num_elements:]

        # Map these indices back to the original order
        top_20_ids = target_ids[top_20_indices]
        top_20_memberships = true_memberships[top_20_indices]
        top_20_log_p_r = self.log_p_r[top_20_indices]
        top_20_log_p_s = self.log_p_s[top_20_indices]

        return top_20_log_p_s, top_20_log_p_r, top_20_ids, top_20_memberships

    def get_true_memberships(self):
        return self.challenger.target_memberships

    def get_adversary_raw_dataset(self, **kwargs):
        return AdversaryRawDataSet(self.adversary_ids, **kwargs)

    def get_adversary_raw_dataset_val(self, **kwargs):
        return AdversaryRawDataSet(self.non_adversary_ids[:int(0.2 * len(self.non_adversary_ids))], **kwargs)


class AdversaryDOMIASDigits(AdversaryDOMIAS):
    def get_adversary_raw_dataset(self, **kwargs):
        return AdversaryDigitsDataSet(self.adversary_ids, **kwargs)

    def get_adversary_raw_dataset_val(self, **kwargs):
        return AdversaryDigitsDataSet(self.non_adversary_ids[:int(0.2 * len(self.non_adversary_ids))], **kwargs)


class AdversaryLOGAN(AdversaryDOMIAS):
    """Only uses log p_S(x_t)"""
    def __init__(
        self,
        _challenger: Challenger,
        data_kwargs: dict,
        nsf_train_loader_kwargs: dict,
        background_knowledge: float = 0.0
    ):
        super().__init__(_challenger, data_kwargs, nsf_train_loader_kwargs, background_knowledge)

    def infer_membership_logx(self, gamma):
        densities = self.f_logx_normalized(self.log_p_s)
        infered_memberships = []
        for d in densities:
            if d >= gamma:
                b = 1
            else:
                b = 0
            infered_memberships.append(b)
        return infered_memberships

    @staticmethod
    def f_logx_normalized(logx, log_b=None):
        """f(logx): [min_diff:max_diff] -> [0, 1]"""

        return (logx - np.min(logx)) / (np.max(logx) - np.min(logx))


class AdversaryZCalibratedDOMIAS(AdversaryDOMIAS):
    """Using p_S(x)/p_R(z)."""
    def __init__(
        self,
        _challenger: Challenger,
        data_kwargs: dict,
        nsf_train_loader_kwargs: dict,
        background_knowledge: float = 0.0,
        n_z=455,
    ):
        self.n_z = n_z
        super().__init__(_challenger, data_kwargs, nsf_train_loader_kwargs, background_knowledge)
        assert len(self.target_ids) == n_atlas, "Z-calibrated DOMIAS needs to use all datasets as targets."

    def infer_membership_logx(self, gamma=1):
        # TODO: try varying beta and gamma
        n_calibration = self.n_z  # TODO: when implementing experiment with different values
        random.seed(20)  # TODO: when implementing, set fresh seed again
        membership_scores = []
        beta_threshold = 0.5
        target_ids = np.array(self.target_ids).copy()

        for ii, target_id in enumerate(target_ids):
            log_p_s_target = np.repeat(self.log_p_s[ii], n_calibration)  # 1-dim array of log_p_s values
            reference_points = random.sample(self.adversary_ids, n_calibration)  # get n_calibration reference points
            # print("ok")
            indices = np.array([np.where(np.array(target_ids == z))[0][0] for z in reference_points])
            log_p_r_z = self.log_p_r[indices]

            diffs = self.f_logx_normalized(log_p_s_target, log_p_r_z)

            membership_scores.append(np.sum((diffs >= gamma)) / n_calibration)

        inferred_membership = np.where(np.array(membership_scores) > beta_threshold, 1, 0)
        return inferred_membership


class AdversaryZCalibratedDOMIAS2(AdversaryZCalibratedDOMIAS):
    """Using p_S(x)/p_S(z)."""
    def __init__(
        self,
        _challenger: Challenger,
        data_kwargs: dict,
        nsf_train_loader_kwargs: dict,
        background_knowledge: float = 0.0,
        n_z=455,
    ):
        self.thresholds = np.arange(0.4, 0.072, 0.001)
        super().__init__(_challenger, data_kwargs, nsf_train_loader_kwargs, background_knowledge, n_z)

    def calculate_log_densities(self):
        nsf_syn = OptimizedNSF(self.synthetic_train_loader, self.synthetic_val_loader)
        syn_saving_keys = {"adversary": "DOMIAS", "syn": "", "sha": f"{subset_to_sha256_key(self.challenger_ids)}"}

        if nsf_syn.model_exists(**syn_saving_keys):
            nsf_syn.load(**syn_saving_keys)
        else:
            nsf_syn.train()
            nsf_syn.save(**syn_saving_keys)

        # for every target x get p_S(x) from the nsf-density estimator
        log_p_s = []
        for target_id in self.target_ids:
            target = self.auxiliary_ds.getitem_by_id(target_id)
            target = np.expand_dims(target, axis=0)
            target = torch.from_numpy(target).to(device=nsf_syn.device)
            # print("log_p_s val:", nsf_syn.eval_log_density(target).item())
            log_p_s.append(nsf_syn.eval_log_density(target).item())

        if "cuda" in str(nsf_syn.device):
            torch.cuda.empty_cache()

        return np.array(log_p_s, dtype=np.float64), np.array(log_p_s, dtype=np.float64)


class AdversaryLiRA:
    def __init__(
            self,
            _challenger: Challenger,
            offline: bool = True,
            n_reference_models: int = 2,  # reference models for in or out world
            background_knowledge: float = 0.0,
            shadow_model_train_loader_kwargs: dict = None,
    ):
        if test_mode:
            random.seed(10)
        else:
            random.seed()
        self.N = n_reference_models
        self.offline = offline
        self.target_ids = _challenger.target_ids
        self.true_memberships = _challenger.target_memberships
        self.n_dataset = len(_challenger.challenger_ids)
        self.synthetic_images_challenger = _challenger.challenger_syn_dataset
        self.m_c = self.synthetic_images_challenger.get_data().shape[0]  # number of synthetic images of the challenger
        self.background_knowledge = background_knowledge

        self.raw_data_kwargs = _challenger.raw_data_kwargs
        self.training_vqvae_kwargs = _challenger.training_vqvae_kwargs
        self.vqvae_train_loader_kwargs = _challenger.vqvae_train_loader_kwargs
        self.training_transformer_kwargs = _challenger.training_transformer_kwargs
        self.shadow_model_train_loader_kwargs = shadow_model_train_loader_kwargs

        # remove targets from id list to avoid unintended (double) membership.
        self.challenger_ids = _challenger.challenger_ids
        self.non_challenger_ids = _challenger.non_challenger_ids
        self.challenger_ids_without_targets = [id for id in self.challenger_ids if id not in self.target_ids]
        self.non_challenger_ids_without_targets = [id for id in self.non_challenger_ids if id not in self.target_ids]
        random.shuffle(self.non_challenger_ids_without_targets)

        # sample adversary shadow dataset ids
        self.adversary_ids = self.sample_shadow_ids(background_knowledge=self.background_knowledge, online=not offline)
        self.all_synthetic_reference_datasets = []

        if offline:
            self.likelihood_out = self.offline_attack()
            self.tprs, self.fprs = None, None
        else:
            self.likelihood_in, self.likelihood_out = self.online_attack()
            self.tprs, self.fprs, thresholds = self.infer_membership_online()

    def infer_membership_online(self):
        likelihood_ratios = np.divide(self.likelihood_in, self.likelihood_out)
        print("likelihood ratios: ", likelihood_ratios)
        return roc_curve(np.array(self.true_memberships), likelihood_ratios)

    def online_attack(self):
        q_out = np.empty((self.N, self.m_c))
        q_in = np.empty((self.N, self.m_c))
        for model_id in range(0, self.N * 2, 2):
            out_ids = self.adversary_ids[model_id, :]
            out_syn_train_loader = self.create_reference_synthetic_dataset(out_ids, model_id, label=0)
            out_reference_model = self.train_reference_model(out_syn_train_loader, out_ids)
            q_out[model_id, :] = self.predict(out_reference_model).ravel()  # all labels are the same

            in_ids = self.adversary_ids[model_id, :]
            in_syn_train_loader = self.create_reference_synthetic_dataset(in_ids, model_id, label=1)
            in_reference_model = self.train_reference_model(in_syn_train_loader, in_ids)
            q_in[model_id, :] = self.predict(in_reference_model).ravel()

        q_in_mean, q_in_std = np.mean(q_in), np.std(q_in)
        q_out_mean, q_out_std = np.mean(q_out), np.std(q_out)

        # get "confidence" scores for target synthetic dataset by training a "global" classifier on all datasets
        global_data_set = ConcatDataset(self.all_synthetic_reference_datasets)
        global_train_loader = get_train_loader(global_data_set, **self.shadow_model_train_loader_kwargs)
        global_model = self.train_reference_model(global_train_loader)
        target_confidence = self.predict(global_model)
        assert len(target_confidence) == self.m_c

        # calculate likelihoods:
        mu_out = np.repeat(q_out_mean, self.m_c)  # Mean vector for non-member distribution
        sigma_out = q_out_std * np.eye(self.m_c)
        mvn_out = multivariate_normal(mean=mu_out, cov=sigma_out)  # multivariate spherical normal
        likelihood_out = mvn_out.pdf(target_confidence)

        mu_in = np.repeat(q_in_mean, self.m_c)  # Mean vector for non-member distribution
        sigma_in = q_in_std * np.eye(self.m_c)
        mvn_in = multivariate_normal(mean=mu_in, cov=sigma_in)
        likelihood_in = mvn_in.pdf(target_confidence)

        return likelihood_in, likelihood_out

    def offline_attack(self):
        confidence = np.empty((self.N, self.m_c))
        for model_id in range(self.N):
            out_ids = self.adversary_ids[model_id, :]
            syn_train_loader = self.create_reference_synthetic_dataset(out_ids, model_id)
            reference_model = self.train_reference_model(syn_train_loader, out_ids)

            confidence[model_id, :] = self.predict(reference_model)  # len(log_p_st_given_s) == len(s_target)

        # get mean and std of losses:
        mu = np.mean(confidence)
        std = np.std(confidence)

        if True:
            import matplotlib.pyplot as plt
            plt.hist(confidence, color='blue', alpha=0.8, bins=100)
            plt.xlabel("loss")
            plt.show()

        mean_vector = np.repeat(mu, self.m_c)
        covariance_matrix = std * np.eye(self.m_c)
        raise NotImplementedError

    def create_reference_synthetic_dataset(self, ids, model_id, label=None):
        sha_key = "3cc5fefe" if (test_mode and label == 0) else subset_to_sha256_key(ids)
        # sha_key = subset_to_sha256_key(ids)
        ds = AdversaryRawDataSet(ids, **self.raw_data_kwargs)
        adversary_train_loader = get_train_loader(ds, **self.vqvae_train_loader_kwargs)

        # train challenger model, train VQVAE and transformer decoder at once using a fixed seed
        t_vqvae = train_transformer_and_vqvae(adversary_train_loader,
                                              vqvae_kwargs=self.training_vqvae_kwargs,
                                              transformer_kwargs=self.training_transformer_kwargs,
                                              saving_kwargs={
                                                  "adversary": str(model_id),
                                                  "epochs": f"{self.training_vqvae_kwargs['n_epochs']}_"
                                                            f"{self.training_transformer_kwargs['n_epochs']}",
                                                  "stopping": f"{self.training_vqvae_kwargs['early_stopping_patience']}_"
                                                              f"{self.training_transformer_kwargs['early_stopping_patience']}",
                                                  "sha": sha_key,
                                                  "l": label}
                                              )

        # load m synthetic datapoints or generate and save them
        path_to_syn_data = os.path.join(self.training_transformer_kwargs["model_path"], "synthetic_data")
        syn_data_file = os.path.join(path_to_syn_data,
                                        f"adversary{model_id}_syn_knowledge{self.background_knowledge}_"
                                        f"sha{sha_key}_l{label}.npy")

        if not os.path.isfile(path_to_syn_data):
            challenger_syn_imgs = t_vqvae.create_synthetic_images(self.m_c)

            if not os.path.exists(path_to_syn_data):
                os.makedirs(path_to_syn_data)

            np.save(syn_data_file, challenger_syn_imgs)

        if label is None:
            syn_data_set = SyntheticDataSet(path_to_syn_data)
        else:
            labels = np.repeat(label, self.m_c)
            syn_data_set = SyntheticDataSet(path_to_syn_data, labels)

        if "cuda" in str(t_vqvae.device):
            torch.cuda.empty_cache()

        self.all_synthetic_reference_datasets.append(syn_data_set)

        return get_train_loader(syn_data_set, **self.shadow_model_train_loader_kwargs)

    def train_reference_model(self, data_loader, ids=None):
        """Train reference model on synthetic data."""
        raise NotImplementedError

    def predict(self, model):
        raise NotImplementedError

    @staticmethod
    def logit_transform(logits):
        return logits

    def sample_shadow_ids(self, background_knowledge=0.0, online=False):
        """Sample N datasets where the targets are evenly distributed such that one target is N/2 of the dataset and not
        in the other N/2-half of the datasets. Return list of indices representing the datapoints that the adversary
        uses to train its shadow models.

        choices background_knowledge: 0.0, 0.5, 1.0
        The overlap is never at 0% (however it is less than 5%) since some targets which are member of the challenger ds
        are distributed among the datasets of the adversary. This is to ensure the parallelization of the attack.

        The overlap is also never at 100% (but probably more than 90%) since the random assigment of targets may
        disproportionately overwrite either challenger or non-challenger ids.
        """
        assert max(self.non_challenger_ids_without_targets) < 65535, "Range of ids incompatible with uint16"

        n_datasets = self.N * 2 if online else self.N
        step_size = 2 if online else 1

        datasets = np.empty((n_datasets, self.n_dataset), dtype=np.uint16)

        for ii in range(0, n_datasets, step_size):
            row = []
            if background_knowledge == 0.0:
                # sample N random datasets on non_challenger ids without any target, fill the rest with challenger_ids
                row = self.non_challenger_ids_without_targets.copy()
                row.extend(random.sample(self.challenger_ids_without_targets, self.n_dataset - len(row)))

            elif background_knowledge == 0.5:
                # for every shadow dataset take 50% of adversary ids and 50% of non-adversary ids
                assert self.n_dataset % 2 == 0, "Size of dataset should be divisible by 2."
                row = random.sample(self.non_challenger_ids_without_targets, self.n_dataset // 2)
                row.extend(random.sample(self.challenger_ids_without_targets, self.n_dataset // 2))

            elif background_knowledge == 1.0:
                # sample N random datasets on challenger ids without any targets, always use full list of challenger ids
                # challenger_ids_without target is too small to fully fill the adversary datasets with n_dataset samples
                # beware that the overlap is never 100% because
                row = self.challenger_ids_without_targets.copy()

                # fill the rest of the adversary datasets with non_challenger_ids
                row.extend(random.sample(self.non_challenger_ids_without_targets, self.n_dataset - len(row)))
            else:
                print("this amount of adversary knowledge is not handled yet")

            if online:
                in_ids = row.copy()
                in_ids[len(row) - len(self.target_ids):] = self.target_ids  # replacing last ids in list with targets
                random.shuffle(in_ids)
                assert len(in_ids) == len(set(in_ids)), f"duplicates: {set([x for x in in_ids if in_ids.count(x) > 1])}"
                datasets[ii + 1, :] = in_ids

            random.shuffle(row)
            assert len(row) == len(set(row)), f"{set([x for x in row if row.count(x) > 1])} are duplicates"
            assert len(row) == self.n_dataset
            datasets[ii, :] = row

            # print("number of duplicates in row: ", np.sum(np.unique(row, return_counts=True)[1] - 1))

        return datasets

    def get_empirical_overlap(self):
        """Calculate the mean (and std) of the overlap between the adversary datasets and the challenger dataset.
        Calculate with targets."""
        overlap = np.empty(self.adversary_ids.shape[0])
        for ii in range(self.adversary_ids.shape[0]):
            adversary_list = list(self.adversary_ids[ii, :])
            overlap[ii] = len(set(adversary_list) & set(self.challenger_ids)) / len(self.challenger_ids)

        mean = overlap.mean()
        std = overlap.std()

        return mean, std


class AdversaryLiRAClassifier(AdversaryLiRA):
    def __init__(
        self,
        _challenger: Challenger,
        offline: bool = True,
        n_reference_models: int = 2,  # reference models for in or out world
        background_knowledge: float = 0.0,
        shadow_model_train_loader_kwargs: dict = None,
        membership_classifier_kwargs: dict = None,
    ):
        self.membership_classifier_kwargs = membership_classifier_kwargs
        super().__init__(_challenger, offline, n_reference_models, background_knowledge,
                         shadow_model_train_loader_kwargs)

    def train_reference_model(self, data_loader, ids=None):
        """Train membership classifier."""
        classifier = MembershipClassifier(data_loader, **self.membership_classifier_kwargs)
        saving_keys = {"adversary": "LiRA_classifier", "syn": "", "sha": f"{subset_to_sha256_key(ids)}"}

        if classifier.model_exists(**saving_keys):
            classifier.load(**saving_keys)
        else:
            classifier.train()
            classifier.save(**saving_keys)

        return classifier

    def predict(self, model, loss=True):
        """Get un-normalized features a.k.a. logits (i.e., the outputs of the model’s last layer before the softmax
        function)"""
        loader = get_train_loader(self.synthetic_images_challenger, batch_size=1, augment_flag=False, num_workers=1)
        logits = []
        for img in loader:
            logits.append(model.logits(img))

        if "cuda" in model.device:
            torch.cuda.empty_cache()

        return self.logit_transform(logits)

    @staticmethod
    def logit_transform(logits):
        """See Carlini et al. 2022. Using all classes because the class of the target is not known."""
        logits = np.array(logits)

        # compute the LogSumExp of all logits for numerical stability
        max_logit = np.max(logits)
        logsumexp_all = np.log(np.sum(np.exp(logits - max_logit))) + max_logit

        # calculate phi(f(x)_y) for each class in a numerically stable way
        return logits - logsumexp_all


class AdversaryLiRANSF(AdversaryLiRA):
    def train_reference_model(self, data_loader, ids=None):
        """Train NSF."""
        assert ids is not None
        nsf = OptimizedNSF(data_loader)
        saving_keys = {"adversary": "LiRA_NSF", "syn": "", "sha": f"{subset_to_sha256_key(ids)}"}

        if nsf.model_exists(**saving_keys):
            nsf.load(**saving_keys)
        else:
            nsf.train()
            nsf.save(**saving_keys)

        return nsf

    def predict(self, model):
        loader = get_train_loader(self.synthetic_images_challenger, batch_size=1, augment_flag=False, num_workers=1)
        log_p_s = []
        for batch in loader:
            # print("img.shape: ", img.shape)
            img = batch
            log_p_s.append(model.eval_log_density(img).cpu())
        return np.array(log_p_s)


class AdversaryRMIANSF(AdversaryLiRANSF):
    def __init__(
            self,
            _challenger: Challenger,
            nsf_train_loader_kwargs: dict,
            offline: bool = True,
            n_reference_models: int = 2,  # reference models for in or out world
            background_knowledge: float = 0.0,
    ):
        self.nsf_train_loader_kwargs = nsf_train_loader_kwargs
        super().__init__(_challenger, offline, n_reference_models, background_knowledge)

    def infer_membership(self):
        raise NotImplementedError


class MultiSeedAdversaryDOMIAS(AdversaryDOMIAS):
    def __init__(
        self,
        log_p_s,
        log_p_r,
        true_memberships,
        _challenger: Challenger,
        data_kwargs: dict,
        nsf_train_loader_kwargs: dict,
        background_knowledge: float = 0.0,
        outlier_percentile: float = None,
    ):

        self.log_p_s = log_p_s
        self.log_p_r = log_p_r
        self._true_memberships = true_memberships
        super().__init__(_challenger, data_kwargs, nsf_train_loader_kwargs, background_knowledge, outlier_percentile)

        print("true_memberships len: ", len(self.true_memberships))
        print("self.log_p_s len: ", len(self.log_p_s))
        if outlier_percentile is not None:
            self.log_p_s, self.log_p_r, self.target_ids, self.true_memberships = self.select_outliers(outlier_percentile)


        self.diffs = self.f_logx_normalized(self.log_p_s, self.log_p_r)
        self.fprs, self.tprs, _ = roc_curve(self.true_memberships, self.diffs)
        self.auc = auc(self.fprs, self.tprs)
        print("auc sklearn: ", calculate_AUC(self.tprs, self.fprs))
        print("auc (sklearn) sklearn: ", auc(self.fprs, self.tprs))

        #self.diffs = self.log_p_s - self.log_p_r
        #self.tprs, self.fprs = self.roc_curve()

        #print("auc own: ", calculate_AUC(self.tprs, self.fprs))
        #print("auc (sklearn) own: ", auc(self.fprs, self.tprs))



    def calculate_log_densities(self):
            return self.log_p_s, self.log_p_r

    def get_true_memberships(self):
        return self._true_memberships
