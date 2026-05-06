import os
import cv2
import json
import numpy as np
import torch
import colorsys
from torch.utils import data

# ==========================================

# ==========================================
def mix_chroma(mixmap, chroma_list, illum_count):
    """
    完全复刻参考代码的混合逻辑
    """
    ret = np.zeros_like(mixmap, dtype=np.float32)
    for i in range(len(illum_count)):
        illum_idx = int(illum_count[i]) - 1
        weight = mixmap[:, :, i:i+1]
        color = chroma_list[illum_idx].reshape(1, 1, 3)
        ret += weight * color
    return ret

def worker_init_fn(worker_id):
    cv2.setNumThreads(0)
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# ==========================================

# ==========================================
class RandomColor():
    def __init__(self, sat_min, sat_max, val_min, val_max, hue_threshold):
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.hue_threshold = hue_threshold

    def hsv2rgb(self, h, s, v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))
    
    def threshold_test(self, hue_list, hue):
        if len(hue_list) == 0: return True
        for h in hue_list:
            if abs(h - hue) < self.hue_threshold: return False
        return True

    def __call__(self, illum_count):
        hue_list = []
        ret_chroma = [[0,0,0],[0,0,0],[0,0,0]]
        for i in illum_count:
            while(True):
                hue = np.random.uniform(0, 1)
                saturation = np.random.uniform(self.sat_min, self.sat_max)
                value = np.random.uniform(self.val_min, self.val_max)
                
                
                chroma_rgb = np.array(self.hsv2rgb(hue, saturation, value), dtype='float32')
                
                
                
                if chroma_rgb[1] < 1e-5:
                    chroma_rgb[1] = 1e-5
                
                
                chroma_rgb /= chroma_rgb[1]

                if self.threshold_test(hue_list, hue):
                    hue_list.append(hue)
                    ret_chroma[int(i)-1] = chroma_rgb
                    break
        return np.array(ret_chroma)


class RandomRotateFlip:
    """
    训练模式专属数据增强：仅对train集生效
    1. 随机旋转：角度范围 [-30°, 30°]
    2. 随机翻转：50%概率水平翻转 + 50%概率垂直翻转
    3. 所有指定张量【严格同步增强】，保证空间位置完全对齐
    4. 支持numpy数组 (H,W,C) 格式，与原数据格式完美兼容
    """
    def __init__(self, split):
        self.split = split
        self.aug_keys = ["input", "input_rgb", "gt", "gt_rgb", "gt_illum", "mask", "mixmap"]
        self.rot_min = -30  
        self.rot_max = 30   

    def __call__(self, ret_dict):
        
        if self.split != 'train' and np.random.random() > 0.7  :
            return ret_dict
        # print('RandomRotateFlip Augmentation Applied')
        
        random_angle = np.random.uniform(self.rot_min, self.rot_max)  
        flip_horizontal = np.random.random() > 0.5                    
        flip_vertical = np.random.random() > 0.5                      
        
        for key in self.aug_keys:
            if key not in ret_dict:
                continue
            data = ret_dict[key]
            if not isinstance(data, np.ndarray):
                continue
            
            data = self.rotate_np(data, random_angle)
            
            if flip_horizontal:
                data = np.flip(data, axis=1)
            
            if flip_vertical:
                data = np.flip(data, axis=0)
            
            ret_dict[key] = data
        return ret_dict

    def rotate_np(self, img_np, angle):
        """numpy数组旋转封装，适配任意通道数 (H,W,C)"""
        h, w = img_np.shape[:2]
        center = (w / 2, h / 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = cv2.warpAffine(
            img_np, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
        
        if len(img_np.shape) == 3 and len(rotated.shape) == 2:
            rotated = rotated[..., np.newaxis]
        return rotated

   
# ==========================================

# ==========================================
class IllumDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        illum_mode: str,
        split_json_path: str,
        image_pool: list = [1, 2, 3],
        output_type: str = "illum", 
        transform=None,
        uncalculable: int = -1,
        mask_black: float = 0.0,
        illum_augmentation = None
    ):
        self.root = root
        self.split = split
        self.illum_mode = illum_mode
        self.split_json_path = split_json_path
        self.image_pool = image_pool
        self.output_type = output_type
        self.transform = transform
        self.uncalculable = uncalculable
        self.mask_black = mask_black

        self.allowed_prefixes = self._load_prefixes()
        self.target_suffixes = self._get_target_suffixes()
        self.image_list = self._generate_image_list()

        self.illum_augmentation = illum_augmentation
        
        if 'galaxy' in self.root.lower():
            self.norm_divisor = 1024.0  
        else:
            self.norm_divisor = 16384.0 

    def _load_prefixes(self) -> set:
        if not os.path.exists(self.split_json_path):
            raise FileNotFoundError(f"split.json not found: {self.split_json_path}")
        with open(self.split_json_path, 'r') as f:
            data = json.load(f)
        prefixes = []
        prefixes.extend(data.get(f"two_illum_{self.split}", []))
        prefixes.extend(data.get(f"three_illum_{self.split}", []))
        return set(prefixes)

    def _get_target_suffixes(self) -> list:
        suffix_map = {
            "single": ["_1"],
            "multi": ["_12", "_13", "_123"],
            "mixed": ["_1", "_12", "_13", "_123"]
        }
        valid_suffixes = []
        for suffix in suffix_map[self.illum_mode]:
            count = len(suffix.replace("_", ""))
            if count in self.image_pool:
                valid_suffixes.append(suffix)
        return valid_suffixes

    def _generate_image_list(self) -> list:
        data_dir = os.path.join(self.root, self.split)
        if not os.path.exists(data_dir): return []
        valid_images = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".tiff") and "gt" not in fname:
                name_no_ext = os.path.splitext(fname)[0]
                parts = name_no_ext.split('_')
                if len(parts) < 2: continue
                prefix = parts[0]
                illum_suffix = "_" + parts[-1]
                if prefix in self.allowed_prefixes and illum_suffix in self.target_suffixes:
                    valid_images.append(fname)
        return valid_images

    def _load_mask(self, data_dir, prefix, shape_hw):
        mask_path = os.path.join(data_dir, f"{prefix}_mask.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32)
            if mask.shape != shape_hw:
                mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.ones(shape_hw, dtype=np.float32)
        return mask[..., None]

    def __getitem__(self, idx):
        ret_dict = {}
        img_file = self.image_list[idx]
        data_dir = os.path.join(self.root, self.split)
        fname_no_ext = os.path.splitext(img_file)[0]
        prefix, illum_ids = fname_no_ext.split('_')
        
        ret_dict["img_file"] = img_file
        ret_dict["place"] = prefix
        ret_dict["illum_count"] = illum_ids

        # 1. Load Input & GT RGB
        raw_path = os.path.join(data_dir, img_file)
        raw_bgr = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / self.norm_divisor

        gt_path = os.path.join(data_dir, f"{fname_no_ext}_gt.tiff")
        gt_bgr = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / self.norm_divisor

        h, w = raw_rgb.shape[:2]
        
        # 2. Load Mixmap & GT Illum
        
        mixmap_valid = np.zeros((h, w, 3), dtype=np.float32)
        full_mixmap = np.zeros((h, w, 3), dtype=np.float32)

        if len(illum_ids) == 1:
            mixmap_valid[..., 0] = 1.0
            full_mixmap[..., 0] = 1.0
        else:
            mixmap_path = os.path.join(data_dir, f"{fname_no_ext}.npy")
            if os.path.exists(mixmap_path):
                mixmap_data = np.load(mixmap_path).astype(np.float32)
                if mixmap_data.ndim == 2: mixmap_data = mixmap_data[..., None]
                mixmap_data = np.nan_to_num(mixmap_data, nan=0.0)
                
                valid_c = min(mixmap_data.shape[2], 3)
                full_mixmap[..., :valid_c] = mixmap_data[..., :valid_c]
                full_mixmap[full_mixmap == self.uncalculable] = 0.0
                
                
                mixmap_valid[..., :valid_c] = np.where(mixmap_data[..., :valid_c] == self.uncalculable, 0, mixmap_data[..., :valid_c])

        
        illum_path = os.path.join(data_dir, f"{fname_no_ext}_illum.npy")
        if os.path.exists(illum_path):
            gt_illum_full = np.load(illum_path).astype(np.float32)
            if gt_illum_full.ndim == 2: gt_illum_full = gt_illum_full[..., None]
            if gt_illum_full.shape[:2] != (h, w):
                 gt_illum_full = cv2.resize(gt_illum_full, (w, h), interpolation=cv2.INTER_LINEAR)
            gt_illum_full = np.nan_to_num(gt_illum_full, nan=0.0)
        else:
            gt_illum_full = np.zeros((h, w, 3), dtype=np.float32)


        
        # if self.split == 'train' and self.illum_augmentation is not None:
        
        
        #     augment_chroma = self.illum_augmentation(illum_ids)
            
        
        #     tint_map = mix_chroma(mixmap_valid, augment_chroma, illum_ids)
            
        
        #     # Input_New = Input_Old * Tint
        #     raw_rgb = raw_rgb * tint_map
            
        #     # GT_New = GT_Old * Tint
        
        
        #     gt_illum_full = gt_illum_full * tint_map


        # 4. Load Mask & Apply
        mask = self._load_mask(data_dir, prefix, (h, w))
        if self.mask_black > 0:
             mask[raw_rgb[..., 1:2] < 1e-4] = 0.0

        raw_rgb = raw_rgb * mask
        gt_rgb = gt_rgb * mask
        gt_illum_full = gt_illum_full * mask

        # 5. Output Formatting (RGB -> RB)
        if gt_illum_full.shape[2] == 3:
            gt_illum = np.delete(gt_illum_full, 1, axis=2) 
        else:
            gt_illum = gt_illum_full

        ret_dict["input_rgb"] = raw_rgb
        ret_dict["gt_rgb"] = gt_rgb
        ret_dict["gt_illum"] = gt_illum
        ret_dict["mask"] = mask
        ret_dict["mixmap"] = full_mixmap
        
        # Alias
        ret_dict["input"] = ret_dict["input_rgb"]
        ret_dict["gt"] = ret_dict["gt_illum"] if self.output_type == "illum" else ret_dict["gt_rgb"]

        if self.transform:
            ret_dict = self.transform(ret_dict)

        return ret_dict
        
    def __len__(self):
        return len(self.image_list)

class ToTensor:
    def __call__(self, ret_dict: dict) -> dict:
        keys_to_process = ["input", "input_rgb", "gt", "gt_rgb", "gt_illum", "mask", "mixmap"]
        for key in keys_to_process:
            if key in ret_dict and isinstance(ret_dict[key], np.ndarray):
                data = ret_dict[key]
                if data.ndim == 2: data = data[..., None]
                if data.ndim == 3: data = data.transpose((2, 0, 1))
                ret_dict[key] = torch.from_numpy(data.copy()).float()
        return ret_dict