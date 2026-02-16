import os, random
from PIL import Image
from itertools import chain
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from itertools import islice
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
        caption_file=("caption_2k.txt", "caption_5k.txt"),
        sample_folders=[
            ('Mul_samples_10_2K', 'Mul_samples_30_2K', 'Mul_samples_50_2K'),
            ('Mul_samples_40_5K', 'Mul_samples_50_5K')
        ],
        train=True,
        full_set=False,
        transform="224",
        num_samples_per_caption=3,
        train_size=6000,
        test_size=1000,
        k_shot=None,                     
    ):
        self.root = root
        self.caption_file = caption_file
        self.sample_folders = list(sample_folders)
        self.train = train
        self.transform = transform
        self.num_samples_per_caption = num_samples_per_caption
        self.k_shot = k_shot           
        self._caption_offset = 0
        self.train_size = train_size
        self.test_size = test_size
        self.full_set = full_set

        self.__load_captions__()
        self.__build_images__()

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
        self.captions = []
        
        for idx, caption_filename in enumerate(self.caption_file):
            caption_path = os.path.join(self.root, caption_filename)
            
            with open(caption_path, 'r', encoding='utf-8') as f:
                captions = [line.strip() for line in f if line.strip()]
                self.captions.extend(captions)
        
        if not self.captions:
            raise FileNotFoundError(f"No valid caption files found in {self.root}")

    def __numeric_sort(self, paths):
        return sorted(paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("rohit_caption_", "")))
    
    def __collect_group_captions__(self, sample_folders):
        per_caption = []
        remaining = self.num_samples_per_caption

        for folder in sample_folders:
            if remaining <= 0:
                break

            parts = folder.split("_")
            folder_max = int(parts[2])  

            take = min(folder_max, remaining)

            for sample_idx in range(1, take + 1):
                sample_dir = os.path.join(self.root, folder, str(sample_idx))
                if not os.path.exists(sample_dir):
                    raise RuntimeError(f"Missing directory: {sample_dir}")

                image_files = []
                if "2K" in folder:
                    max_caption_index = 1999
                elif "5K" in folder:
                    max_caption_index = 4999
                else:
                    raise ValueError(f"Unknown folder type: {folder}")

                for f in os.listdir(sample_dir):
                    if not f.lower().endswith(".png"):
                        continue

                    base = os.path.splitext(f)[0]
                    idx = int(base.replace("rohit_caption_", ""))

                    if 0 <= idx <= max_caption_index:
                        image_files.append(os.path.join(sample_dir, f))

                image_files = self.__numeric_sort(image_files)

                for caption_i, img_path in enumerate(image_files):
                    if len(per_caption) <= caption_i:
                        per_caption.append([])
                    per_caption[caption_i].append(img_path)

            remaining -= take

        return per_caption

    def __build_images__(self):
        self.images = []
        self.labels = []

        all_per_caption_images = []

        for sample_folder_tuple in self.sample_folders:
            group_per_caption = self.__collect_group_captions__(sample_folder_tuple)
            all_per_caption_images.extend(group_per_caption)

        total = len(all_per_caption_images)
        train_end = min(self.train_size, total)
        test_end  = min(self.train_size + self.test_size, total)

        if self.full_set:
            selected = all_per_caption_images
            caption_offset = 0
        elif self.train:
            selected = all_per_caption_images[:train_end]
            caption_offset = 0
        else:
            selected = all_per_caption_images[train_end:test_end]
            caption_offset = train_end

        # 🔍 DEBUG / VALIDATION SECTION
        missing_total = 0
        for idx, imgs in enumerate(selected):
            if len(imgs) != self.num_samples_per_caption:
                diff = self.num_samples_per_caption - len(imgs)
                missing_total += max(diff, 0)
                print(
                    f"[WARNING] Caption {idx + caption_offset} "
                    f"has {len(imgs)} images (missing {diff})"
                )

        if missing_total > 0:
            print(f"\nTOTAL MISSING IMAGES: {missing_total}\n")

        for caption_idx, imgs in enumerate(selected):
            imgs = imgs[:self.num_samples_per_caption]
            for img_path in imgs:
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