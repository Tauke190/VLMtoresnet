import os, random
from PIL import Image
from itertools import chain
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC


def choose_resolution(n_px):
    n_px = n_px.split("/")
    if len(n_px) > 1:
        return int(random.choice(n_px))
    else:
        return int(n_px[0])

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def few_shot_resolution_transform(n_px, n_px_org=224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        Resize(n_px_org, interpolation=BICUBIC),
        CenterCrop(n_px_org),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


class DiffisionImages(Dataset):

    def __init__(
        self,
        root="./datasets/Diffision_images",
        caption_file="caption_2k.txt",
        sample_folders=('Mul_samples_10_2K', 'Mul_samples_30_2K', 'Mul_samples_50_2K'),
        train=True,
        transform="224",
        num_samples_per_caption=3,
        k_shot=None,                     
    ):
        self.root = root
        self.caption_file = caption_file
        self.sample_folders = list(sample_folders)
        self.train = train
        self.transform = transform
        self.num_samples_per_caption = num_samples_per_caption
        self.k_shot = k_shot           

        self.__load_captions__()
        self.__build_image_paths__()

        if self.k_shot is not None:
            if self.k_shot == -1:
                print("Not in few-shot setting")
            else:
                self.fewshot()         

        self.__setup_transform__()

    def fewshot(self):
        selected_images, selected_labels = [], []
        unique_classes = set(self.labels)

        for class_label in unique_classes:
            class_indices = [i for i in range(len(self.images)) if self.labels[i] == class_label]
            if len(class_indices) >= self.k_shot:
                chosen = random.sample(class_indices, self.k_shot)
            else:
                chosen = class_indices

            selected_images.extend([self.images[i] for i in chosen])
            selected_labels.extend([self.labels[i] for i in chosen])

        self.images = selected_images
        self.labels = selected_labels

    def __load_captions__(self):
        caption_path = os.path.join(self.root, self.caption_file)
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Caption file not found: {caption_path}")

        with open(caption_path, 'r', encoding='utf-8') as f:
            self.captions = [line.strip() for line in f if line.strip()]

    def __numeric_sort(self, paths):
        return sorted(paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("rohit_caption_", "")))

    def __build_image_paths__(self):
        self.images = []
        self.labels = []
        parts = self.sample_folders[0].split("_")
        num_samples = int(parts[2])

        for sample_idx in range(1, num_samples + 1):
            per_folder_lists = []

            for folder_name in self.sample_folders:
                sample_dir = os.path.join(self.root, folder_name, str(sample_idx))
                if not os.path.exists(sample_dir):
                    raise RuntimeError(f"Missing directory: {sample_dir}")

                image_files = [
                    os.path.join(sample_dir, f)
                    for f in os.listdir(sample_dir)
                    if f.lower().endswith(".png")
                ]

                image_files = self.__numeric_sort(image_files)
                per_folder_lists.append(image_files)

            aligned = list(zip(*per_folder_lists))

            for caption_idx, aligned_imgs in enumerate(aligned):
                for img_path in aligned_imgs:
                    self.images.append(img_path)
                    self.labels.append(caption_idx)

        assert len(self.images) == len(self.labels)

    def __setup_transform__(self):
        if isinstance(self.transform, str):
            self.trans = None
        else:
            self.trans = self.transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.k_shot is not None:
            self.trans = few_shot_resolution_transform(n_px=choose_resolution(self.transform))

        img = Image.open(self.images[index])
        if self.trans is not None:
            img = self.trans(img)
        label = self.labels[index]
        return img, label