import os
import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
import cv2

def build_data_dict(images_dir, masks_dir):
    """Crea un diccionario compatible con el DataLoader original a partir de imágenes PNG."""
    img_files = sorted(os.listdir(images_dir))
    data_dict = {
        'img_npy': [os.path.join(images_dir, f) for f in img_files],
        'anno_npy': [os.path.join(masks_dir, f) for f in img_files],
        'patient_id': [f.split("_patch")[0].split("_neg")[0] for f in img_files]
    }
    return data_dict

def get_train_transform(patch_size, prob=0.5):
    tr_transforms = [
        SpatialTransform(
            patch_size,
            [i // 2 for i in patch_size],
            do_elastic_deform=True, alpha=(0., 300.), sigma=(20., 40.),
            do_rotation=True, angle_x=(-np.pi/15., np.pi/15.),
            angle_y=(-np.pi/15., np.pi/15.), angle_z=(0., 0.),
            do_scale=True, scale=(1/1.15, 1.15),
            random_crop=False,
            border_mode_data='constant', border_cval_data=0,
            order_data=3,
            p_el_per_sample=prob, p_rot_per_sample=prob, p_scale_per_sample=prob
        ),
        MirrorTransform(axes=(1,)),
        BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=prob),
        GaussianNoiseTransform(noise_variance=(0, 0.5), p_per_sample=prob),
        GaussianBlurTransform(blur_sigma=(0.5, 2.0), different_sigma_per_channel=True,
                               p_per_channel=prob, p_per_sample=prob),
        ContrastAugmentationTransform(contrast_range=(0.75, 1.25), p_per_sample=prob)
    ]
    return Compose(tr_transforms)

class GastricDataLoader(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded,
                 crop_status=True, crop_type="random", margins=(0,0,0),
                 seed_for_shuffle=1234, return_incomplete=False, shuffle=True, infinite=True,
                 tr_transforms=None):
        super().__init__(data, batch_size, num_threads_in_multithreaded,
                         seed_for_shuffle, return_incomplete, shuffle, infinite)
        self.patch_size = patch_size
        self.n_channel = 3
        self.indices = list(range(len(data['img_npy'])))
        self.crop_status = crop_status
        self.crop_type = crop_type
        self.margins = margins
        self.tr_transforms = tr_transforms
    @staticmethod
    def load_image(img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img

    @staticmethod
    def load_mask(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = np.expand_dims(mask > 127, axis=-1)  # binarizar
        return mask

    def generate_train_batch(self):
        idx = self.get_indices()
        gland_img = [self._data['img_npy'][i] for i in idx]
        img_seg = [self._data['anno_npy'][i] for i in idx]
        patient_id = [self._data['patient_id'][i] for i in idx]

        img = np.zeros((len(gland_img), self.n_channel, *self.patch_size), dtype=np.float32)
        seg = np.zeros((len(img_seg), 1, *self.patch_size), dtype=np.float32)

        for i, (j, k) in enumerate(zip(gland_img, img_seg)):
            img_data = self.load_image(j)
            seg_data = self.load_mask(k)

            img_data = np.einsum('hwc->chw', img_data)
            seg_data = np.einsum('hwc->chw', seg_data)

            if self.crop_status:
                img_data, seg_data = crop(img_data[None], seg=seg_data[None],
                                          crop_size=self.patch_size,
                                          margins=self.margins,
                                          crop_type=self.crop_type)
                img[i] = img_data[0]
                seg[i] = seg_data[0]
            else:
                pass
            # 🔥 Aquí aplicas los aumentos si están definidos
            if self.tr_transforms is not None:
                augmented = self.tr_transforms(data=img_data, seg=seg_data)
                img_data = augmented['data'][0]  # (3, H, W)
                seg_data = augmented['seg'][0]  # (1, H, W)

            img[i] = img_data
            seg[i] = seg_data
        return {'data': img, 'seg': seg, 'patient_id': patient_id}

    def __len__(self):
        import math
        return math.ceil(len(self.indices) / self.batch_size)
