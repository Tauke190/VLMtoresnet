from timm.data.dataset_factory import create_dataset as timm_create_dataset
from CLIP.dataloaders import DiffisionImages


def create_dataset(
    name,
    root,
    split="train",
    is_training=False,
    class_map=None,
    download=False,
    batch_size=None,
    repeats=0,
    **kwargs,
):

    if name.lower() == "diffusion":

        train = split == "train"
        val = split in ("validation", "val")

        dataset = DiffisionImages(
            root=root,
            train=train,
            val=val,
            full_set=False,
        )

        return dataset

    return timm_create_dataset(
        name,
        root=root,
        split=split,
        is_training=is_training,
        class_map=class_map,
        download=download,
        batch_size=batch_size,
        repeats=repeats,
        **kwargs,
    )