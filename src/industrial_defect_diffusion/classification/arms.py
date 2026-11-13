"""Deterministic full training batches with class balance and exact source quotas."""

from collections import defaultdict
from collections.abc import Iterator
from random import Random

from torch.utils.data import DataLoader, Dataset, Sampler

from ..data.preprocessing import NEUManifestDataset
from ..data.synthetic import ApprovedSyntheticDataset

Pool = NEUManifestDataset | ApprovedSyntheticDataset


class ArmDataset(Dataset):
    """Combine only training pools and identify the source of each exposure."""

    split = "train"

    def __init__(self, pools: dict[str, Pool]) -> None:
        self.pools = pools
        self.indices = []
        self.by_class = {}
        for source, pool in pools.items():
            if pool.split != "train":
                raise ValueError("Training arm requires train-only pools")
            groups = defaultdict(list)
            for index, row in enumerate(pool.rows):
                groups[row["class_id"]].append(len(self.indices))
                self.indices.append((source, index))
            if set(groups) != set(range(6)):
                raise ValueError(f"{source} training pool must contain all six classes")
            self.by_class[source] = groups

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        source, offset = self.indices[index]
        return {**self.pools[source][offset], "source": source}


class ArmBatchSampler(Sampler[list[int]]):
    """Cycle shuffled class orders within each source and sample images with replacement."""

    def __init__(self, dataset: ArmDataset, quotas: dict[str, int], steps: int, seed: int) -> None:
        self.dataset = dataset
        self.quotas = quotas
        self.steps = steps
        self.seed = seed

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = Random(self.seed)
        orders = {source: [] for source in self.quotas}
        for _ in range(self.steps):
            batch = []
            for source, quota in self.quotas.items():
                order = orders[source]
                for _ in range(quota):
                    if not order:
                        order.extend(rng.sample(range(6), 6))
                    label = order.pop()
                    batch.append(rng.choice(self.dataset.by_class[source][label]))
            rng.shuffle(batch)
            yield batch


def make_arm_loader(
    arm: str,
    real: NEUManifestDataset | None,
    synthetic: ApprovedSyntheticDataset | None,
    settings: dict,
    seed: int,
) -> DataLoader:
    """Use only the selected sources and the common optimizer update budget."""
    steps = settings["max_steps"]
    if steps is None:
        raise ValueError("Three-arm comparison requires a shared max_steps budget")
    batch_size = settings["batch_size"]
    if arm == "real_only" and real is not None:
        pools, quotas = {"real": real}, {"real": batch_size}
    elif arm == "synthetic_only" and synthetic is not None:
        pools, quotas = {"synthetic": synthetic}, {"synthetic": batch_size}
    elif arm == "real_plus_synthetic" and real is not None and synthetic is not None:
        synthetic_count = round(batch_size * settings["mixed_synthetic_fraction"])
        pools = {"real": real, "synthetic": synthetic}
        quotas = {"real": batch_size - synthetic_count, "synthetic": synthetic_count}
    else:
        raise ValueError(f"Missing training pool or invalid arm: {arm}")
    dataset = ArmDataset(pools)
    return DataLoader(dataset, batch_sampler=ArmBatchSampler(dataset, quotas, steps, seed))
