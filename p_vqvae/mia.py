from p_vqvae.dataloader import RawDataSet, SyntheticDataSet, get_train_loader
from p_vqvae.networks import train_transformer_and_vqvae
from p_vqvae.neural_spline_flow import OptimizedNSF
from p_vqvae.utils import subset_to_sha256_key
import numpy as np
import random
import torch
import os

# don't forget to clear model outputs folder before implementing
# also check kwargs before implementation
TEST_MODE = True  # turn off for real attack

N = 32  # number of shadow models
n_atlas = 955
n_train = n_atlas // 2  # amount of images for the challenger and the adversary models. ATLAS v2 dataset n = 955

assert N % 2 == 0, "Make sure to give an even N. Else the distribution of targets is uneven."

include_targets_in_reference_dataset = None  # either True, False or None (=included at random)

if TEST_MODE:
    epochs_vqvae = 1
    epochs_transformer = 1
    downsample = 4
    m_syn_images = 2
    n_targets = 5
else:
    epochs_vqvae = 100
    epochs_transformer = 50
    downsample = 1
    m_syn_images = n_train
    n_targets = 100

data_kwargs = {
    "root": "/home/chrsch/P_VQVAE/data/ATLAS_2",
    "cache_path": '/home/chrsch/P_VQVAE/data/cache/',
    "downsample": downsample,
    "normalize": 1,
    "crop": ((8, 9), (12, 13), (0, 9)),
    "padding": ((1, 2), (0, 0), (1, 2))}
train_loader_kwargs = {
    "batch_size": 1,
    "augment_flag": True,
    "num_workers": 2
}
nsf_train_loader_kwargs = {
    "batch_size": 1,
    "augment_flag": False,
    "num_workers": 1
}
training_vqvae_kwargs = {
    "n_epochs": epochs_vqvae,
    "multiple_devices": False,
    "dtype": torch.float32,
    "use_checkpointing": True,
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
    "lr": 5e-4,
    "model_path": "/home/chrsch/P_VQVAE/model_outputs/lira"
}


# TODO: Create function that takes model checks if synthetic data already exist. If yes, return mmap. If not create data


class ChallengerRawDataSet(RawDataSet):
    def __init__(self, seed, **kwargs):
        super().__init__(mode="full", **kwargs)
        self.challenger_ids, self.non_challenger_ids = self.sample_challenger_dataset(seed)

    def __getitem__(self, idx):
        """Overwrite __get_item__ method, so that the idx refers to index within the challenger dataset. Maps input
        "idx" to items in the challenger dataset."""
        i = self.challenger_ids[idx]  # mapping of idx to item in challenger dataset
        image = np.copy(self.data[i])

        if self.transform:
            image = self.transform(image)

        return {"image": image}

    def __len__(self):
        """Return length of challenger dataset instead of the size of the whole data distribution."""
        return len(self.challenger_ids)

    def sample_challenger_dataset(self, seed):
        """Take entire (memory mapped) dataset and sample n_train images. Return sampled dataset in random order and the
        rest of the dataset distribution without the challenger dataset."""
        random.seed(seed)  # fixed seed for sampling datasets
        random_challenger_ids = random.sample(range(0, n_atlas), n_train)
        non_challenger_ids = [i for i in range(0, n_atlas) if i not in random_challenger_ids]

        # return distribution[random_challenger_ids], distribution[non_challenger_ids]
        return random_challenger_ids, non_challenger_ids


class Challenger:
    def __init__(self, m_c):
        self.m_c = m_c
        self.data_seed = 420
        self.training_seed = 69

        # sample challenger dataset using fixed seed
        challenger_ds = ChallengerRawDataSet(self.data_seed, **data_kwargs)
        challenger_train_loader = get_train_loader(challenger_ds, **train_loader_kwargs)

        # train challenger model, train VQVAE and transformer decoder at once using a fixed seed
        t_vqvae = train_transformer_and_vqvae(challenger_train_loader, training_vqvae_kwargs,
                                              training_transformer_kwargs,
                                              saving_kwargs={"challenger": 1, "seed": self.training_seed},
                                              seed=self.training_seed)

        # load m synthetic datapoints or generate and save them
        path_to_syn_data = os.path.join(training_transformer_kwargs["model_path"], "synthetic_data")
        path_to_syn_data = os.path.join(path_to_syn_data, f"challenger_syn_seed{self.training_seed}_2.npy")

        if not os.path.isfile(path_to_syn_data):
            challenger_syn_imgs = t_vqvae.create_synthetic_images(self.m_c)
            np.save(path_to_syn_data, challenger_syn_imgs)

        self.challenger_syn_dataset = SyntheticDataSet(path_to_syn_data)

        # self.n_c = challenger_ds.__len__()  # size of challenger dataset
        self.challenger_ids = challenger_ds.challenger_ids
        self.non_challenger_ids = challenger_ds.non_challenger_ids
        self.target_ids, self.target_memberships = self.sample_n_random_target_ids(n_targets)

    def sample_n_random_target_ids(self, _n):
        """Sample N targets at once. Flip random bit b. If if b = 1: choose target in challenger dataset. If b = 0:
         choose t not in dataset and remove target from non-challenger data to avoid unintended membership.
         Return indices of targets and their membership label."""
        random.seed()  # fresh seed

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

        return {"image": image}

    def __len__(self):
        """Return length of challenger dataset instead of the size of the whole data distribution."""
        return len(self.adversary_ids)

    def getitem_by_id(self, id):
        image = np.copy(self.data[id])
        return {"image": image}

class AdversaryDOMIAS:
    def __init__(
            self,
            _challenger: Challenger,
            background_knowledge: float = 0.0
    ):
        self.synthetic_train_loader = get_train_loader(_challenger.challenger_syn_dataset, **nsf_train_loader_kwargs)
        self.background_knowledge = background_knowledge
        self.challenger_ids = _challenger.challenger_ids
        self.non_challenger_ids = _challenger.non_challenger_ids
        self.target_ids = _challenger.target_ids
        self.true_memberships = _challenger.target_memberships
        self.adversary_ids = self.sample_adversary_ids(n=len(self.challenger_ids))
        self.adversary_ds = AdversaryRawDataSet(self.adversary_ids, **data_kwargs)
        self.adversary_train_loader = get_train_loader(self.adversary_ds, **nsf_train_loader_kwargs)

        # train model on raw and on synthetic dataset and calculate log densities for each target
        self.log_p_S, self.log_p_R = self.calculate_log_densities()
        self.infer_membership_f_x()
        # infer membership p_S/P_R = exp(log_P_S - log_P_R)

    def sample_adversary_ids(self, n):
        random.seed()  # fresh seed

        if include_targets_in_reference_dataset is None:
            challenger_ids = self.challenger_ids.copy()
            non_challenger_ids = self.non_challenger_ids.copy()
        else:
            raise NotImplementedError

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

    def train_nsf(self, train_ds):
        nsf = OptimizedNSF(train_ds)

    def calculate_log_densities(self):
        nsf_syn = OptimizedNSF(self.synthetic_train_loader)
        syn_saving_keys = {"adversary":"DOMIAS", "syn":"", "sha":f"{subset_to_sha256_key(self.challenger_ids)}"}
        if nsf_syn.model_exists(**syn_saving_keys):
            nsf_syn.load(**syn_saving_keys)
        else:
            nsf_syn.train(**syn_saving_keys)
            nsf_syn.save(**syn_saving_keys)

        # for every target x get p_S(x) from the nsf-density estimator
        log_p_S = []
        for target_id in self.target_ids:
            target = self.adversary_ds.getitem_by_id(target_id)['image']
            target = np.expand_dims(target, axis=0)
            target = torch.from_numpy(target).to(device=nsf_syn.device)
            log_p_S.append(nsf_syn.eval_log_density(target).item())

        if "cuda" in str(nsf_syn.device):
            torch.cuda.empty_cache()

        nsf_raw = OptimizedNSF(self.adversary_train_loader)
        raw_saving_keys = {"adversary": "DOMIAS", "raw": "", "sha": f"{subset_to_sha256_key(self.adversary_ids)}"}
        if nsf_raw.model_exists(**raw_saving_keys):
            nsf_raw.load(**raw_saving_keys)
        else:
            nsf_raw.train(**raw_saving_keys)
            nsf_raw.save(**raw_saving_keys)

        log_p_R = []
        for target_id in self.target_ids:
            target = self.adversary_ds.getitem_by_id(target_id)['image']
            target = np.expand_dims(target, axis=0)
            target = torch.from_numpy(target).to(device=nsf_raw.device)
            log_p_R.append(nsf_raw.eval_log_density(target).item())

        # p_S/P_R = exp(p_S - p_R)
        print("log_p_S: ", log_p_S)
        print("log_p_R: ", log_p_R)
        return np.array(log_p_S, dtype=np.float64), np.array(log_p_R, dtype=np.float64)

    def infer_membership_f_x(self, gamma=1):
        diffs = self.log_p_S - self.log_p_R
        print("diff: ", diffs)
        infered_memberships = []
        for diff in diffs:
            if diff >= gamma:
                b = 1
            else:
                b = 0
            infered_memberships.append(b)
        return infered_memberships
        # ratio = np.exp(self.log_p_S - self.log_p_R)

    def roc_curve(self):
        for gamma in range(-20000, 0, 1000):
            inferred_memberships = self.infer_membership_f_x()

            tp = np.sum((self.true_memberships == 1) & (inferred_memberships == 1))
            tn = np.sum((self.true_memberships == 0) & (inferred_memberships == 0))
            fp = np.sum((self.true_memberships == 0) & (inferred_memberships == 1))
            fn = np.sum((self.true_memberships == 1) & (inferred_memberships == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            #fpr = fp






class AdversaryLiRA:
    def __init__(
            self,
            _challenger: Challenger,
            background_knowledge: float = 0.0,
    ):

        self.target_ids = _challenger.target_ids
        self.n_dataset = len(_challenger.challenger_ids)
        self.synthetic_images_challenger = _challenger.challenger_syn_dataset
        self.m_c = self.synthetic_images_challenger.get_data().shape[0]  # number of synthetic images of the challenger
        self.background_knowledge = background_knowledge

        # remove targets from id list to avoid unintended (double) membership.
        self.challenger_ids = _challenger.challenger_ids
        self.non_challenger_ids = _challenger.non_challenger_ids
        self.challenger_ids_without_targets = [id for id in self.challenger_ids if not id in self.target_ids]
        self.non_challenger_ids_without_targets = [id for id in self.non_challenger_ids if not id in self.target_ids]
        random.shuffle(self.non_challenger_ids_without_targets)

        # sample adversary shadow dataset ids
        self.adversary_ids = self.sample_shadow_ids(background_knowledge=self.background_knowledge)

        # for every target, define an in- or out-dataset.
        # TODO: should training begin in this class?
        # self.adversary_datasets = self.get_adversary_datasets(**dataset_kwargs)

    def sample_shadow_ids(self, background_knowledge=0.0):
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
        datasets = np.empty((N, self.n_dataset), dtype=np.uint16)

        for ii in range(N):
            row = []
            if background_knowledge == 0.0:
                # sample N random datasets on non_challenger ids without any target
                row = random.sample(self.non_challenger_ids_without_targets, self.n_dataset)

            elif background_knowledge == 0.5:
                # for every shadow dataset take 50% of adversary ids and 50% of non-adversary ids
                assert self.n_dataset % 2 == 0, "Size of dataset should be divisible by 2."
                row = random.sample(self.non_challenger_ids_without_targets, self.n_dataset // 2)
                row.extend(random.sample(self.challenger_ids_without_targets, self.n_dataset // 2))
                random.shuffle(row)

            elif background_knowledge == 1.0:
                # sample N random datasets on challenger ids without any targets, always use full list of challenger ids
                # challenger_ids_without target is too small to fully fill the adversary datasets with n_dataset samples
                # beware that the overlap is never 100% because
                row = self.challenger_ids_without_targets.copy()

                # fill the rest of the adversary datasets with non_challenger_ids
                row.extend(random.sample(self.non_challenger_ids_without_targets, self.n_dataset - len(row)))
                random.shuffle(row)
            else:
                print("this amount of adversary knowledge is not handled yet")

            assert len(row) == len(set(row)), f"{set([x for x in row if row.count(x) > 1])} are duplicates"
            assert len(row) == self.n_dataset
            datasets[ii, :] = row

        # distribute targets, make sliding window assign half of each column with one target
        starting_k = 0
        ending_k = N // 2
        ending_k_2 = 1  # needed if sliding window reaches the border (where ending_k >= N).

        for ii in range(N):
            column = datasets[:, ii].copy()  # get colum
            single_target_id = self.target_ids[ii]  # select target

            if ending_k <= N:
                column[starting_k:ending_k] = np.repeat(single_target_id, N//2)
            else:
                column[0: ending_k_2] = np.repeat(single_target_id, ending_k_2)
                column[starting_k:N] = np.repeat(single_target_id, N - starting_k)
                ending_k_2 = +1

            # shift assignment sliding window by 1 for the next column:
            starting_k += 1
            ending_k += 1

            datasets[:, ii] = column  # re-assign column

        return datasets

    def get_empirical_overlap(self):
        """Calculate the mean (and std) of the overlap between the adversary datasets and the challenger dataset.
        Calculate with targets."""
        overlap = np.empty(N)
        for ii in range(N):
            adversary_list = list(self.adversary_ids[ii, :])
            overlap[ii] = len(set(adversary_list) & set(self.challenger_ids)) / len(self.challenger_ids)

        mean = overlap.mean()
        std = overlap.std()

        return mean, std

    def get_adversary_datasets(self, target, **kwargs) -> list[RawDataSet]:
        datasets = []
        # for N in datasets:
        #    pass
        return datasets

    def train_shadow_generators(self, datasets, **training_kwargs):
        """Train N shadow model generators. Save their weights in a file."""
        shadow_models = []

        return shadow_models

    def create_synthetic_datasets(self, m_a):
        """For each model create m_a synthetic shadow datasets. Save them in the same folder as the models. If data
        exist, load from disc. Return mmap of synthetic datasets"""
        synthetic_images_adversary = []

        return synthetic_images_adversary

    def train_shadow_classifier(self):
        """For every target t redefine S_in, S_out and train a classifier f on the membership of t."""
        shadow_classifiers = []
        return shadow_classifiers

    def logit_scaling(self):
        pass

    def train_global_classifier(self):
        pass

    def calculate_likelihood_ratio(self):
        pass


if __name__ == "__main__":
    challenger = Challenger(3)
    print(challenger.challenger_ids)
    adversary = AdversaryDOMIAS(challenger, background_knowledge=1.0)
    print(challenger.target_ids)
    print(challenger.target_memberships)
    # adversary = AdversaryLiRA(challenger)
