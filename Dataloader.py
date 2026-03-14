
import random
import numpy as np
from PIL import Image, ImageFile

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from utility import RandomIdentitySampler, RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

ImageFile.LOAD_TRUNCATED_IMAGES = True

__factory = {
    'Mars': Mars,
    'iLIDSVID': iLIDSVID,
    'PRID': PRID
}


class TrackletCamStyleAugment:
    """Generator-free, tracklet-consistent camera-style augmentation.

    This is *CamStyle-inspired* rather than the original CycleGAN-based CamStyle.
    It applies one shared photometric style transform to all frames in the sampled
    tracklet so temporal consistency is preserved.

    Input tensor is expected to be normalized to [-1, 1].
    """

    def __init__(self, probability=0.5, strength=0.18, noise_std=0.01):
        self.probability = float(probability)
        self.strength = float(strength)
        self.noise_std = float(noise_std)

    def _sample_params(self, device, dtype):
        s = self.strength
        brightness = 1.0 + random.uniform(-0.8, 0.8) * s
        contrast = 1.0 + random.uniform(-0.8, 0.8) * s
        saturation = 1.0 + random.uniform(-0.7, 0.7) * s
        gamma = 1.0 + random.uniform(-0.6, 0.6) * s
        channel_gain = torch.tensor(
            [1.0 + random.uniform(-0.6, 0.6) * s for _ in range(3)],
            device=device,
            dtype=dtype,
        ).view(1, 3, 1, 1)
        channel_bias = torch.tensor(
            [random.uniform(-0.10, 0.10) * s for _ in range(3)],
            device=device,
            dtype=dtype,
        ).view(1, 3, 1, 1)
        return brightness, contrast, saturation, gamma, channel_gain, channel_bias

    def __call__(self, imgs, camid=None):
        if random.random() >= self.probability:
            return imgs

        x = imgs.mul(0.5).add(0.5).clamp(0.0, 1.0)
        brightness, contrast, saturation, gamma, channel_gain, channel_bias = self._sample_params(
            x.device, x.dtype
        )

        # Brightness
        x = x * brightness

        # Saturation around grayscale image
        gray = x.mean(dim=1, keepdim=True)
        x = gray + saturation * (x - gray)

        # Contrast around per-frame mean intensity
        frame_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = frame_mean + contrast * (x - frame_mean)

        # Camera-dependent color response shift
        x = x * channel_gain + channel_bias
        x = x.clamp(1e-6, 1.0)

        # Gamma curve shift
        x = x.pow(gamma)

        # Mild sensor noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * (self.noise_std * self.strength)
            x = (x + noise).clamp(0.0, 1.0)

        return x.sub(0.5).div(0.5)


def train_collate_fn(batch):
    imgs, pids, camids, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    labels = torch.stack(labels, dim=0)
    return imgs, pids, camids, labels


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    camids = torch.tensor(camids, dtype=torch.int64)
    return imgs, pids, camids, img_paths


def dataloader(
    Dataset_name,
    batch_size=64,
    num_workers=4,
    seq_len=4,
    use_camstyle_aug=True,
    camstyle_prob=0.5,
    camstyle_strength=0.18,
):
    train_transforms = T.Compose([
        T.Resize([256, 128], interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = __factory[Dataset_name]()
    camstyle_aug = TrackletCamStyleAugment(probability=camstyle_prob, strength=camstyle_strength) if use_camstyle_aug else None

    train_set = VideoDataset_inderase(
        dataset.train,
        seq_len=seq_len,
        sample='intelligent',
        transform=train_transforms,
        camstyle_aug=camstyle_aug,
    )

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=RandomIdentitySampler(dataset.train, batch_size, 4),
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
    )

    q_val_set = VideoDataset(
        dataset.query,
        seq_len=seq_len,
        sample='dense',
        transform=val_transforms,
    )
    g_val_set = VideoDataset(
        dataset.gallery,
        seq_len=seq_len,
        sample='dense',
        transform=val_transforms,
    )

    q_val_loader = DataLoader(
        q_val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True,
    )

    g_val_loader = DataLoader(
        g_val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True,
    )

    return train_loader, len(dataset.query), num_classes, cam_num, view_num, q_val_loader, g_val_loader


def read_image(img_path):
    """Keep reading image until succeed."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def _pad_indices(indices, seq_len):
    indices = list(indices)
    if len(indices) == 0:
        raise RuntimeError("Empty tracklet encountered.")
    while len(indices) < seq_len:
        indices.append(indices[-1])
    return indices


class VideoDataset(Dataset):
    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len)
            begin_index = random.randint(0, rand_end)
            indices = frame_indices[begin_index:begin_index + self.seq_len]
            indices = _pad_indices(indices, self.seq_len)

            imgs = []
            targt_cam = []
            for idx in indices:
                img_path = img_paths[int(idx)]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
                targt_cam.append(camid)

            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, targt_cam

        elif self.sample == 'dense':
            frame_indices = list(range(num))
            indices_list = []

            cur_index = 0
            while cur_index + self.seq_len <= num:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len

            if cur_index < num:
                last_seq = frame_indices[cur_index:]
                last_seq = _pad_indices(last_seq, self.seq_len)
                indices_list.append(last_seq)

            if len(indices_list) == 0:
                indices_list.append(_pad_indices(frame_indices, self.seq_len))

            imgs_list = []
            for indices in indices_list[:self.max_length]:
                imgs = []
                for idx in indices:
                    img_path = img_paths[int(idx)]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    imgs.append(img.unsqueeze(0))
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)

            imgs_array = torch.stack(imgs_list, dim=0)
            return imgs_array, pid, camid, img_paths

        elif self.sample == 'dense_subset':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.max_length)
            begin_index = random.randint(0, rand_end) if rand_end > 0 else 0

            indices_list = []
            cur_index = begin_index
            while cur_index + self.seq_len <= num:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len

            if cur_index < num:
                last_seq = frame_indices[cur_index:]
                last_seq = _pad_indices(last_seq, self.seq_len)
                indices_list.append(last_seq)

            if len(indices_list) == 0:
                indices_list.append(_pad_indices(frame_indices, self.seq_len))

            imgs_list = []
            for indices in indices_list[:self.max_length]:
                imgs = []
                for idx in indices:
                    img_path = img_paths[int(idx)]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    imgs.append(img.unsqueeze(0))
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)

            imgs_array = torch.stack(imgs_list, dim=0)
            return imgs_array, pid, camid

        elif self.sample == 'intelligent_random':
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    left = min(i * each, num - 1)
                    right = min((i + 1) * each - 1, num - 1)
                else:
                    left = min(i * each, num - 1)
                    right = num - 1
                indices.append(random.randint(left, right))

            imgs = []
            for idx in indices:
                img_path = img_paths[int(idx)]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, camid

        else:
            raise KeyError("Unknown sample method: {}".format(self.sample))


class VideoDataset_inderase(Dataset):
    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40, camstyle_aug=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.camstyle_aug = camstyle_aug
        self.erase = RandomErasing3(probability=0.5)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample != "intelligent":
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len)
            begin_index = random.randint(0, rand_end)
            indices = frame_indices[begin_index:begin_index + self.seq_len]
            indices = _pad_indices(indices, self.seq_len)
        else:
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    left = min(i * each, num - 1)
                    right = min((i + 1) * each - 1, num - 1)
                else:
                    left = min(i * each, num - 1)
                    right = num - 1
                indices.append(random.randint(left, right))

        imgs = []
        targt_cam = []
        for idx in indices:
            img_path = img_paths[int(idx)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img.unsqueeze(0))
            targt_cam.append(camid)

        imgs = torch.cat(imgs, dim=0)

        # Apply one shared camera-style perturbation to the whole sampled tracklet.
        if self.camstyle_aug is not None:
            imgs = self.camstyle_aug(imgs, camid=camid)

        labels = []
        erased_imgs = []
        for frame in imgs:
            frame = frame.clone()
            frame, temp = self.erase(frame)
            labels.append(temp)
            erased_imgs.append(frame.unsqueeze(0))

        labels = torch.tensor(labels)
        imgs = torch.cat(erased_imgs, dim=0)
        return imgs, pid, targt_cam, labels
