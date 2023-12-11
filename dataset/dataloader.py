import os,glob
import torch
import cv2
import scipy
import string
import json
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np

from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from os.path import join as ospj
from torchvision.utils import save_image
from random import choice, randint, sample, uniform, shuffle
from util import *


def region_draw_text(H, W, r_bbox, text, font_path = "./dataset/utils/arial.ttf"):

    m_top, m_bottom, m_left, m_right = r_bbox
    m_h, m_w = m_bottom-m_top, m_right-m_left

    font = ImageFont.truetype(font_path, 128)
    std_l, std_t, std_r, std_b = font.getbbox(text)
    std_h, std_w = std_b - std_t, std_r - std_l
    image = Image.new('RGB', (std_w, std_h), color = (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, fill = (0, 0, 0), font=font, anchor="lt")
    
    transform = transforms.Compose([
        transforms.Resize((m_h, m_w), transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor()
    ])
    
    image = transform(image)

    result = torch.ones((3, H, W))
    result[:, m_top:m_bottom, m_left:m_right] = image

    return result


def initialize_word_dict():

        with open('./dataset/utils/words.txt', 'r') as f:
            word_list = f.readlines()
        
        words = []
        for word_line in word_list:
            words += word_line[:-1].split(" ")

        words.sort(key = lambda w: len(w))
        word_dict = {l:[] for l in range(len(words[0]), len(words[-1])+1)}
        for word in words:
            word_dict[len(word)].append(word)

        return word_dict
        

class LabelDataset(data.Dataset):

    def __init__(self, size, length, font_path, min_len, max_len) -> None:
        super().__init__()

        # constraint
        self.length = length
        self.size = size

        # path
        self.font_path = font_path

        # word dict
        self.character = string.printable[:-6]
        self.min_len = min_len
        self.max_len = max_len

        self.grayscale = transforms.Grayscale()
        self.resize = transforms.Resize((self.size, self.size), transforms.InterpolationMode.BICUBIC, antialias=True)

    def __len__(self):
        
        return self.length
    
    def __getitem__(self, index):

        while True:

            text_len = randint(self.min_len, self.max_len)
            text = "".join([choice(self.character) for i in range(text_len)])
            font_path = self.font_path

            try: 
                font = ImageFont.truetype(font_path, 128)
                std_l, std_t, std_r, std_b = font.getbbox(text)
                std_h, std_w = std_b - std_t, std_r - std_l
                if std_h == 0 or std_w == 0:
                    continue
            except:
                continue
            
            try:
                image = Image.new('RGB', (std_w, std_h), color = (0,0,0))
                draw = ImageDraw.Draw(image)
                draw.text((0, 0), text, fill = (255,255,255), font=font, anchor="lt")
            except:
                continue

            image = transforms.ToTensor()(image)
            image = self.grayscale(image)
            image = self.resize(image)

            batch = {
                "image": image,
                "text": text
            }

            return batch
    

class ICDAR13Dataset(data.Dataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__()

        # basic
        self.type = datype
        self.character = string.printable[:-6]

        # path
        self.data_root = ospj(cfgs.data_root, "ICDAR13", self.type)
        self.image_root = ospj(self.data_root, "images")
        self.anno_root = ospj(self.data_root, "annos")
        self.anno_paths = sorted(glob.glob(ospj(self.anno_root, "*.txt")))

        # constraint
        self.H = cfgs.H
        self.W = cfgs.W
        self.word_len = cfgs.word_len
        self.seq_len = cfgs.seq_len
        self.mask_min_ratio = cfgs.mask_min_ratio
        self.aug_text_enabled = cfgs.aug_text_enabled
        self.aug_text_ratio = cfgs.aug_text_ratio

        self.items = []
        total_count = 0
        for anno_path in self.anno_paths:
            name = anno_path.split(os.sep)[-1].split(".")[0].replace("gt_", "")
            with open(anno_path, "r") as fp:
                annos = fp.readlines()

            for anno in annos:

                total_count += 1
                text = anno.split("\"")[1]
                left, top, right, bottom = [int(s) for s in anno.split(", ")[:4]]
                area = (bottom-top) * (right-left)
                bbox = np.array((top, bottom, left, right))

                if len(text) < self.word_len[0] or len(text) > self.word_len[1]: continue
                if not all([c in self.character for c in text]): continue
                if area / (self.H * self.W) < self.mask_min_ratio: continue

                self.items.append({
                    "image_path": ospj(self.image_root, f"{name}.jpg"),
                    "text": text,
                    "bbox": bbox
                })

        self.length = len(self.items)
        print(f"Total: {total_count}, filtered: {self.length}")
        self.count = -1
        self.word_dict = initialize_word_dict()
    
    def __len__(self):
        
        return self.length
    
    def augment(self, image, bbox):

        h, w, _ = image.shape
        m_top, m_bottom, m_left, m_right = bbox

        mask = np.ones((h, w), dtype=np.uint8)
        mask[m_top:m_bottom, m_left:m_right] = 0

        if h >= w:
            delta = (h-w)//2
            m_left += delta; m_right += delta
            image = cv2.copyMakeBorder(image, 0,0,delta,delta, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, 0,0,delta,delta, cv2.BORDER_CONSTANT, value = (1,1,1))
        else:
            delta = (w-h)//2
            m_top += delta; m_bottom += delta
            image = cv2.copyMakeBorder(image, delta,delta,0,0, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, delta,delta,0,0, cv2.BORDER_CONSTANT, value = (1,1,1))

        m_h, m_w = int(m_bottom-m_top), int(m_right-m_left)
        c_h, c_w = m_top + m_h//2, m_left + m_w//2

        h, w, _ = image.shape
        area = (m_bottom-m_top) * (m_right-m_left)
        aug_min_ratio = self.mask_min_ratio * 4
        if area/(h*w) < aug_min_ratio:
            d = int((area/aug_min_ratio)**0.5)
            d = max(d, max(m_h, m_w))
            if c_h <= h - c_h:
                delta_top = min(c_h, d//2)
                delta_bottom = d - delta_top
            else:
                delta_bottom = min(h - c_h, d//2)
                delta_top = d - delta_bottom
            if c_w <= w - c_w:
                delta_left = min(c_w, d//2)
                delta_right = d - delta_left
            else:
                delta_right = min(w - c_w, d//2)
                delta_left = d - delta_right

            n_top, n_bottom = c_h - delta_top, c_h + delta_bottom
            n_left, n_right = c_w - delta_left, c_w + delta_right

            image = image[n_top:n_bottom, n_left:n_right, :]
            mask = mask[n_top:n_bottom, n_left:n_right]

            m_top -= n_top; m_bottom -= n_top
            m_left -= n_left; m_right -= n_left

        h, w, _ = image.shape
        m_top, m_bottom = int(m_top * (self.H/h)), int(m_bottom * (self.H/h))
        m_left, m_right = int(m_left * (self.W/w)), int(m_right * (self.W/w))
        
        image = cv2.resize(image, (self.W, self.H))
        mask = cv2.resize(mask, (self.W, self.H))

        r_bbox = torch.tensor((m_top, m_bottom, m_left, m_right))
        
        return image, mask, r_bbox
        
    def __getitem__(self, index):

        self.count += 1

        item = self.items[index]
        image_path = item["image_path"]
        text = item["text"]
        bbox = item["bbox"]

        aug_text = choice(self.word_dict[len(text)]) if uniform(0, 1) <= self.aug_text_ratio else text

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        image = np.asarray(image)
        image, mask, r_bbox = self.augment(image, bbox)

        image = torch.from_numpy(image.transpose(2,0,1)).to(dtype=torch.float32) / 127.5 - 1.0

        mask = torch.from_numpy(mask[None]).to(dtype=torch.float32)
        masked = image * mask
        mask = 1 - mask

        seg_mask = torch.cat((torch.ones(len(text)), torch.zeros(self.seq_len-len(text))))

        rendered = region_draw_text(self.H, self.W, r_bbox, aug_text if self.aug_text_enabled else text)

        # additional cond
        txt = f"\"{aug_text if self.aug_text_enabled else text}\""
        original_size_as_tuple = torch.tensor((h, w))
        crop_coords_top_left = torch.tensor((0, 0))
        target_size_as_tuple = torch.tensor((self.H, self.W))

        batch = {
            "image": image,
            "mask": mask,
            "masked": masked,
            "seg_mask": seg_mask,
            "r_bbox": r_bbox,
            "rendered": rendered,
            "label": aug_text if self.aug_text_enabled else text,
            "txt": txt,
            "original_size_as_tuple": original_size_as_tuple,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size_as_tuple": target_size_as_tuple,
            "name": str(self.count)
        }

        return batch


class TextSegDataset(data.Dataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__()

        # basic
        self.type = datype
        self.character = string.printable[:-6]

        # path
        self.data_root = ospj(cfgs.data_root, "TextSeg", self.type)
        self.image_root = ospj(self.data_root, "image")
        self.anno_root = ospj(self.data_root, "annotation")

        # constraint
        self.H = cfgs.H
        self.W = cfgs.W
        self.word_len = cfgs.word_len
        self.seq_len = cfgs.seq_len
        self.mask_min_ratio = cfgs.mask_min_ratio
        self.seg_min_ratio = cfgs.seg_min_ratio
        self.aug_text_enabled = cfgs.aug_text_enabled
        self.aug_text_ratio = cfgs.aug_text_ratio

        image_paths = sorted(glob.glob(ospj(self.image_root, "*.jpg")))
        anno_paths = sorted(glob.glob(ospj(self.anno_root, "*.json")))
        seg_paths = sorted([p for p in glob.glob(ospj(self.anno_root, "*.png")) if "eff" not in p])

        self.items = []
        total_count = 0
        for image_path, anno_path, seg_path in zip(image_paths, anno_paths, seg_paths):
            with open(anno_path, "rb") as fp:
                annos = json.load(fp)
            for anno in annos.values():
                total_count += 1
                text = anno["text"]
                chars = [anno["char"][key]["text"] for key in anno["char"]]
                bbox = np.array(anno["bbox"]).reshape((4,2))
                seg_values = [c["mask_value"] for c in anno["char"].values()]
                area = cv2.contourArea(bbox)

                if "".join(chars) != text: continue
                if "#" in text: continue
                if len(text) < self.word_len[0] or len(text) > self.word_len[1]: continue
                if not all([c in self.character for c in text]): continue
                if area / (self.H * self.W) < self.mask_min_ratio: continue

                self.items.append({
                    "image_path": image_path,
                    "seg_path": seg_path,
                    "text": text,
                    "bbox": bbox,
                    "seg_values": seg_values
                })

        self.length = len(self.items)
        print(f"Total: {total_count}, filtered: {self.length}")
        self.count = -1
        self.word_dict = initialize_word_dict()

    def __len__(self):
        
        return self.length
    
    def augment(self, image, seg, text, bbox, seg_values):

        h, w, _ = image.shape
        m_top, m_bottom = int(np.min(bbox[:,1])), int(np.max(bbox[:,1]))
        m_left, m_right = int(np.min(bbox[:,0])), int(np.max(bbox[:,0]))

        mask = np.ones((h, w), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, bbox, 0)

        if h >= w:
            delta = (h-w)//2
            m_left += delta; m_right += delta
            image = cv2.copyMakeBorder(image, 0,0,delta,delta, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, 0,0,delta,delta, cv2.BORDER_CONSTANT, value = (1,1,1))
            seg = cv2.copyMakeBorder(seg, 0,0,delta,delta, cv2.BORDER_CONSTANT, value = (0,0,0))
        else:
            delta = (w-h)//2
            m_top += delta; m_bottom += delta
            image = cv2.copyMakeBorder(image, delta,delta,0,0, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, delta,delta,0,0, cv2.BORDER_CONSTANT, value = (1,1,1))
            seg = cv2.copyMakeBorder(seg, delta,delta,0,0, cv2.BORDER_CONSTANT, value = (0,0,0))

        m_h, m_w = int(m_bottom-m_top), int(m_right-m_left)
        c_h, c_w = m_top + m_h//2, m_left + m_w//2

        h, w, _ = image.shape
        area = cv2.contourArea(bbox)
        aug_min_ratio = self.mask_min_ratio * 4
        if area/(h*w) < aug_min_ratio:
            d = int((area/aug_min_ratio)**0.5)
            d = max(d, max(m_h, m_w))
            if c_h <= h - c_h:
                delta_top = min(c_h, d//2)
                delta_bottom = d - delta_top
            else:
                delta_bottom = min(h - c_h, d//2)
                delta_top = d - delta_bottom
            if c_w <= w - c_w:
                delta_left = min(c_w, d//2)
                delta_right = d - delta_left
            else:
                delta_right = min(w - c_w, d//2)
                delta_left = d - delta_right

            n_top, n_bottom = c_h - delta_top, c_h + delta_bottom
            n_left, n_right = c_w - delta_left, c_w + delta_right

            image = image[n_top:n_bottom, n_left:n_right, :]
            mask = mask[n_top:n_bottom, n_left:n_right]
            seg = seg[n_top:n_bottom, n_left:n_right, :]

            m_top -= n_top; m_bottom -= n_top
            m_left -= n_left; m_right -= n_left

        segs = []
        text_indices = [[i for i, c in enumerate(text) if c == ch] for ch in text]
        for i in range(len(text)):
            indices = text_indices[i]
            seg_i = np.sum([(seg == seg_values[ind]).astype(np.uint8).mean(axis=-1) for ind in indices], axis=0) # position un-aware
            seg_i = np.clip(seg_i, 0, 1)
            seg_i = cv2.morphologyEx(seg_i, cv2.MORPH_OPEN, np.ones((1,2),np.int8), iterations=2) # denoise
            seg_i = cv2.morphologyEx(seg_i, cv2.MORPH_OPEN, np.ones((2,1),np.int8), iterations=2) # denoise
            seg_i = cv2.morphologyEx(seg_i, cv2.MORPH_DILATE, np.ones((3,3),np.int8), iterations=7) # dilate
            segs.append(seg_i[None])

        segs = segs + [np.zeros_like(segs[0]) for i in range(self.seq_len-len(segs))]
        seg = np.concatenate(segs, axis=0)

        h, w, _ = image.shape
        m_top, m_bottom = int(m_top * (self.H/h)), int(m_bottom * (self.H/h))
        m_left, m_right = int(m_left * (self.W/w)), int(m_right * (self.W/w))
        
        image = cv2.resize(image, (self.W, self.H))
        seg = cv2.resize(seg.transpose((1,2,0)), (self.W, self.H)).transpose((2,0,1))
        mask = cv2.resize(mask, (self.W, self.H))

        r_bbox = torch.tensor((m_top, m_bottom, m_left, m_right))
        
        return image, seg, mask, r_bbox
        
    def __getitem__(self, index):

        self.count += 1

        while True:

            item = self.items[index]
            image_path = item["image_path"]
            seg_path = item["seg_path"]
            text = item["text"]
            bbox = item["bbox"]
            seg_values = item["seg_values"]
            
            aug_text = choice(self.word_dict[len(text)]) if uniform(0, 1) <= self.aug_text_ratio else text

            image = Image.open(image_path).convert("RGB")
            seg = Image.open(seg_path).convert("RGB")
            w, h = image.size
            image = np.asarray(image)
            seg = np.asarray(seg)
            image, seg, mask, r_bbox = self.augment(image, seg, text, bbox, seg_values)
            
            image = torch.from_numpy(image.transpose(2,0,1)).to(dtype=torch.float32) / 127.5 - 1.0

            mask = torch.from_numpy(mask[None]).to(dtype=torch.float32)
            masked = image * mask
            mask = 1 - mask

            seg = torch.from_numpy(seg)
            seg_mask = torch.cat((torch.ones(len(text)), torch.zeros(self.seq_len-len(text))))

            rendered = region_draw_text(self.H, self.W, r_bbox, aug_text if self.aug_text_enabled else text)

            # additional cond
            txt = f"\"{aug_text if self.aug_text_enabled else text}\""
            original_size_as_tuple = torch.tensor((h, w))
            crop_coords_top_left = torch.tensor((0, 0))
            target_size_as_tuple = torch.tensor((self.H, self.W))

            batch = {
                "image": image,
                "seg": seg,
                "seg_mask": seg_mask,
                "mask": mask,
                "masked": masked,
                "r_bbox": r_bbox,
                "rendered": rendered,
                "label": aug_text if self.aug_text_enabled else text,
                "txt": txt,
                "original_size_as_tuple": original_size_as_tuple,
                "crop_coords_top_left": crop_coords_top_left,
                "target_size_as_tuple": target_size_as_tuple,
                "name": str(self.count)
            }

            return batch


class SynthTextDataset(data.Dataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__()

        # basic
        self.type = datype
        self.length = cfgs.length
        self.character = string.printable[:-6]

        # path
        self.data_root = ospj(cfgs.data_root, "SynthText")
        self.anno_path = ospj(self.data_root, "gt.mat")

        # constraint
        self.H = cfgs.H
        self.W = cfgs.W
        self.word_len = cfgs.word_len
        self.mask_min_ratio = cfgs.mask_min_ratio
        self.seg_min_ratio = cfgs.seg_min_ratio

        anno = scipy.io.loadmat(self.anno_path)
        image_names = anno["imnames"][0]
        word_bboxes = anno["wordBB"][0]
        char_bboxes = anno["charBB"][0]
        txts = anno["txt"][0]

        if cfgs.use_cached:
            with open(ospj(self.data_root, "items.json"), "r") as fp:
                self.items = json.load(fp)
        else:
            self.items = []
            for image_name, word_bbox, char_bbox, txt in zip(image_names, word_bboxes, char_bboxes, txts):
                image_name = image_name[0]
                image_path = ospj(self.data_root, image_name)

                txt_list = []
                for frag in txt:
                    frag = frag.replace("\n", " ")
                    frags = [s for s in frag.split(" ") if s != ""]
                    txt_list += frags
                
                if word_bbox.ndim < 3: word_bbox = word_bbox[...,None]
                word_bbox = word_bbox.transpose((2,1,0)).astype(np.int32)
                char_bbox = char_bbox.transpose((2,1,0)).astype(np.int32)

                pointer = 0
                for bbox, text in zip(word_bbox, txt_list):

                    seg_bboxs = char_bbox[pointer: pointer+len(text)]
                    pointer += len(text)
                    area = cv2.contourArea(bbox)

                    if len(text) < self.word_len[0] or len(text) > self.word_len[1]: continue
                    if area / (self.H * self.W) < self.mask_min_ratio: continue

                    self.items.append({
                        "image_path": image_path,
                        "text": text,
                        "bbox": bbox.tolist(),
                        "seg_bboxs" : seg_bboxs.tolist()
                    })

            with open(ospj(self.data_root, "items.json"), "w") as fp:
                json.dump(self.items, fp)

        self.count = -1
    
    def __len__(self):
        
        return self.length
    
    def augment(self, image, bbox, seg_bboxs):

        h, w, _ = image.shape
        m_top, m_bottom = max(0, int(np.min(bbox[:,1]))), min(h, int(np.max(bbox[:,1])))
        m_left, m_right = max(0, int(np.min(bbox[:,0]))), min(w, int(np.max(bbox[:,0])))

        mask = np.ones((h, w), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, bbox, 0)

        segs = []
        seg_sum = 0
        for seg_bbox in seg_bboxs:
            seg_i = np.zeros_like(mask)
            seg_i = cv2.fillConvexPoly(seg_i, seg_bbox, 1)
            segs.append(seg_i[None])
            seg_sum += seg_i.sum()
        
        seg_ratio = float(seg_sum / len(segs)) / (h*w)
        segs = segs + [np.zeros_like(segs[0]) for i in range(self.word_len[1]-len(segs))]
        seg = np.concatenate(segs, axis=0)

        if h >= w:
            delta = (h-w)//2
            m_left += delta; m_right += delta
            image = cv2.copyMakeBorder(image, 0,0,delta,delta, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, 0,0,delta,delta, cv2.BORDER_CONSTANT, value = (1,1,1))
            seg = cv2.copyMakeBorder(seg.transpose((1,2,0)), 0,0,delta,delta, cv2.BORDER_CONSTANT, value = (0,0,0)).transpose((2,0,1))

        else:
            delta = (w-h)//2
            m_top += delta; m_bottom += delta
            image = cv2.copyMakeBorder(image, delta,delta,0,0, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, delta,delta,0,0, cv2.BORDER_CONSTANT, value = (1,1,1))
            seg = cv2.copyMakeBorder(seg.transpose((1,2,0)), delta,delta,0,0, cv2.BORDER_CONSTANT, value = (0,0,0)).transpose((2,0,1))

        m_h, m_w = int(m_bottom-m_top), int(m_right-m_left)
        c_h, c_w = m_top + m_h//2, m_left + m_w//2

        h, w, _ = image.shape
        area = cv2.contourArea(bbox)
        aug_min_ratio = self.mask_min_ratio * 4
        if area/(h*w) < aug_min_ratio:
            d = int((area/aug_min_ratio)**0.5)
            d = max(d, max(m_h, m_w))
            if c_h <= h - c_h:
                delta_top = min(c_h, d//2)
                delta_bottom = d - delta_top
            else:
                delta_bottom = min(h - c_h, d//2)
                delta_top = d - delta_bottom
            if c_w <= w - c_w:
                delta_left = min(c_w, d//2)
                delta_right = d - delta_left
            else:
                delta_right = min(w - c_w, d//2)
                delta_left = d - delta_right

            n_top, n_bottom = c_h - delta_top, c_h + delta_bottom
            n_left, n_right = c_w - delta_left, c_w + delta_right

            image = image[n_top:n_bottom, n_left:n_right, :]
            mask = mask[n_top:n_bottom, n_left:n_right]
            seg = seg[:, n_top:n_bottom, n_left:n_right]

            m_top -= n_top; m_bottom -= n_top
            m_left -= n_left; m_right -= n_left

        h, w, _ = image.shape
        m_top, m_bottom = int(m_top * (self.H/h)), int(m_bottom * (self.H/h))
        m_left, m_right = int(m_left * (self.W/w)), int(m_right * (self.W/w))
        
        image = cv2.resize(image, (self.W, self.H))
        seg = cv2.resize(seg.transpose((1,2,0)), (self.W, self.H)).transpose((2,0,1))
        mask = cv2.resize(mask, (self.W, self.H))

        r_bbox = torch.tensor((m_top, m_bottom, m_left, m_right))
        
        return image, seg, mask, seg_ratio, r_bbox
        
    def __getitem__(self, index):

        self.count += 1

        while True:
        
            item = choice(self.items)
            image_path = item["image_path"]
            text = item["text"]
            bbox = np.array(item["bbox"])
            seg_bboxs = np.array(item["seg_bboxs"])

            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            image = np.asarray(image)
            image, seg, mask, seg_ratio, r_bbox = self.augment(image, bbox, seg_bboxs)

            if seg_ratio < self.seg_min_ratio: continue
            
            image = torch.from_numpy(image.transpose(2,0,1)).to(dtype=torch.float32) / 127.5 - 1.0

            mask = torch.from_numpy(mask[None]).to(dtype=torch.float32)
            masked = image * mask
            mask = 1 - mask

            seg = torch.from_numpy(seg).to(dtype=torch.float32)
            seg_mask = torch.cat((torch.ones(len(text)), torch.zeros(self.word_len[1]-len(text))))

            # additional cond
            txt = f"\"{text}\""
            original_size_as_tuple = torch.tensor((h, w))
            crop_coords_top_left = torch.tensor((0, 0))
            target_size_as_tuple = torch.tensor((self.H, self.W))

            batch = {
                "image": image,
                "seg": seg,
                "seg_mask": seg_mask,
                "mask": mask,
                "masked": masked,
                "r_bbox": r_bbox,
                "label": text,
                "txt": txt,
                "original_size_as_tuple": original_size_as_tuple,
                "crop_coords_top_left": crop_coords_top_left,
                "target_size_as_tuple": target_size_as_tuple,
                "name": str(self.count)
            }

            return batch


class LAIONOCRDataset(data.Dataset):

    def __init__(self, cfgs, datype) -> None:
        super().__init__()

        # basic
        self.type = datype
        self.character = string.printable[:-6]

        # path
        self.data_root = ospj(cfgs.data_root, "LAION-OCR", self.type)

        # constraint
        self.H = cfgs.H
        self.W = cfgs.W
        self.W_std = 512
        self.H_std = 512
        self.word_len = cfgs.word_len
        self.seq_len = cfgs.seq_len
        self.mask_min_ratio = cfgs.mask_min_ratio
        self.seg_min_ratio = cfgs.seg_min_ratio
        self.aug_text_enabled = cfgs.aug_text_enabled if self.type != "train" else False
        self.aug_text_ratio = cfgs.aug_text_ratio

        if cfgs.use_cached:
            with open(ospj(cfgs.data_root, "LAION-OCR", f"{self.type}_items.json"), "r") as fp:
                self.items = json.load(fp)
        else:
            self.items = []
            data_dirs = sorted(glob.glob(ospj(self.data_root, "*")))
            len_count = area_count = text_count = 0
            for data_dir in data_dirs:
                image_path = ospj(data_dir, "image.jpg")
                ocr_path = ospj(data_dir, "ocr.txt")
                seg_path = ospj(data_dir, "charseg.npy")
                
                with open(ocr_path, "r") as fp:
                    ocrs = fp.readlines()
                for ocr in ocrs:
                    text, bbox_str, _ = ocr.strip("\n").split(" ")
                    bbox = np.array([int(v) for v in bbox_str.split(",")]).reshape((4,2))
                    area = cv2.contourArea(bbox)

                    if len(text) < self.word_len[0] or len(text) > self.word_len[1]:
                        len_count += 1
                        continue
                    if not all([c in self.character for c in text]): 
                        text_count += 1
                        continue
                    if area / (self.W_std*self.H_std) < self.mask_min_ratio:
                        area_count += 1
                        continue

                    self.items.append({
                        "image_path": image_path,
                        "seg_path": seg_path,
                        "text": text,
                        "bbox_str": bbox_str,
                    })

            with open(ospj(cfgs.data_root, "LAION-OCR", f"{self.type}_items.json"), "w") as fp:
                json.dump(self.items, fp)
            
            print(f"Total length: {len(self.items)}  filtered out {len_count} len_ill, {area_count} area_ill, {text_count} text_ill")
            
        self.length = cfgs.length
        self.count = -1
        self.word_dict = initialize_word_dict()

    def __len__(self):
        
        return self.length
    
    def augment(self, image, seg, text, bbox):

        image = cv2.resize(image, (self.W_std, self.H_std))
        seg = cv2.resize(seg.astype(np.uint8), (self.W_std, self.H_std))
        mask = np.ones((self.H_std, self.W_std), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, bbox, 0)

        h, w, _ = image.shape
        m_top, m_bottom = max(0, int(np.min(bbox[:,1]))), min(self.H_std, int(np.max(bbox[:,1])))
        m_left, m_right = max(0, int(np.min(bbox[:,0]))), min(self.W_std, int(np.max(bbox[:,0])))
        m_h, m_w = int(m_bottom-m_top), int(m_right-m_left)
        c_h, c_w = m_top + m_h//2, m_left + m_w//2

        area = cv2.contourArea(bbox)
        aug_min_ratio = self.mask_min_ratio * 4
        if area/(h*w) < aug_min_ratio:
            d = int((area/aug_min_ratio)**0.5)
            d = max(d, max(m_h, m_w))
            if c_h <= h - c_h:
                delta_top = min(c_h, d//2)
                delta_bottom = d - delta_top
            else:
                delta_bottom = min(h - c_h, d//2)
                delta_top = d - delta_bottom
            if c_w <= w - c_w:
                delta_left = min(c_w, d//2)
                delta_right = d - delta_left
            else:
                delta_right = min(w - c_w, d//2)
                delta_left = d - delta_right

            n_top, n_bottom = c_h - delta_top, c_h + delta_bottom
            n_left, n_right = c_w - delta_left, c_w + delta_right

            image = image[n_top:n_bottom, n_left:n_right, :]
            mask = mask[n_top:n_bottom, n_left:n_right]
            seg = seg[n_top:n_bottom, n_left:n_right]

            m_top -= n_top; m_bottom -= n_top
            m_left -= n_left; m_right -= n_left

        seg = seg * (1 - mask)

        segs = [None for i in range(len(text))]
        ch_dict = {}
        for i in range(len(text)):
            if text[i] in ch_dict: ch_dict[text[i]].append(i)
            else: ch_dict[text[i]] = [i]
        
        for ch in ch_dict:
            ind = self.character.find(ch) + 1
            ind_l = self.character.find(ch.lower()) + 1
            seg_i = ((seg == ind).astype(np.uint8) + (seg == ind_l).astype(np.uint8))

            seg_i = cv2.morphologyEx(seg_i, cv2.MORPH_OPEN, np.ones((1,2),np.int8), iterations=1) # denoise
            seg_i = cv2.morphologyEx(seg_i, cv2.MORPH_OPEN, np.ones((2,1),np.int8), iterations=1) # denoise
            seg_i = cv2.morphologyEx(seg_i, cv2.MORPH_DILATE, np.ones((3,3),np.int8), iterations=5) # dilate

            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_i, connectivity=4)
            if retval < len(ch_dict[ch]) + 1:
                return None, None, None, None

            stats = stats[1:].tolist()
            if retval > len(ch_dict[ch]) + 1:
                stats.sort(key = lambda st: st[-1])
                stats.reverse()
                stats = stats[:len(ch_dict[ch])]

            stats.sort(key = lambda st: st[0])
            for idx, stat in enumerate(stats):
                x, y, w, h, s = stat
                s_mask = np.zeros_like(seg_i)
                s_mask[y:y+h, x:x+w] = 1
                seg_i_mask = seg_i * s_mask
                segs[ch_dict[ch][idx]] = seg_i_mask[None]

        segs = segs + [np.zeros_like(segs[0]) for i in range(self.seq_len-len(segs))]
        seg = np.concatenate(segs, axis=0)

        h, w, _ = image.shape
        m_top, m_bottom = int(m_top * (self.H/h)), int(m_bottom * (self.H/h))
        m_left, m_right = int(m_left * (self.W/w)), int(m_right * (self.W/w))

        image = cv2.resize(image, (self.W, self.H))
        seg = cv2.resize(seg.transpose((1,2,0)), (self.W, self.H)).transpose((2,0,1))
        mask = cv2.resize(mask, (self.W, self.H))

        r_bbox = torch.tensor((m_top, m_bottom, m_left, m_right))

        return image, seg, mask, r_bbox
        
    def __getitem__(self, index):
        
        self.count += 1

        while True:
            
            item = choice(self.items)
            image_path = item["image_path"]
            seg_path = item["seg_path"]
            text = item["text"]
            bbox_str = item["bbox_str"]
            bbox = np.array([int(v) for v in bbox_str.split(",")]).reshape((4,2))

            aug_text = choice(self.word_dict[len(text)]) if uniform(0, 1) <= self.aug_text_ratio else text

            image = Image.open(image_path).convert("RGB")
            seg = np.load(seg_path)
            w, h = image.size
            image = np.asarray(image)
            image, seg, mask, r_bbox = self.augment(image, seg, text, bbox)

            if image is None: continue
            
            image = torch.from_numpy(image.transpose(2,0,1)).to(dtype=torch.float32) / 127.5 - 1.0

            mask = torch.from_numpy(mask[None]).to(dtype=torch.float32)
            masked = image * mask
            mask = 1 - mask

            seg = torch.from_numpy(seg).to(dtype=torch.float32)
            seg_mask = torch.cat((torch.ones(len(text)), torch.zeros(self.seq_len-len(text))))

            m_top, m_bottom, m_left, m_right = r_bbox
            ref = image[:, m_top:m_bottom, m_left:m_right]
            ref = F.interpolate(ref[None], (128, 128))[0]

            # rendered = region_draw_text(self.H, self.W, r_bbox, aug_text if self.aug_text_enabled else text)

            # additional cond
            txt = f"\"{aug_text if self.aug_text_enabled else text}\""
            original_size_as_tuple = torch.tensor((h, w))
            crop_coords_top_left = torch.tensor((0, 0))
            target_size_as_tuple = torch.tensor((self.H, self.W))

            batch = {
                "image": image,
                "seg": seg,
                "seg_mask": seg_mask,
                "mask": mask,
                "masked": masked,
                "r_bbox": r_bbox,
                "ref": ref,
                # "rendered": rendered,
                "label": aug_text if self.aug_text_enabled else text,
                "txt": txt,
                "original_size_as_tuple": original_size_as_tuple,
                "crop_coords_top_left": crop_coords_top_left,
                "target_size_as_tuple": target_size_as_tuple,
                "name": str(self.count)
            }

            return batch


def get_dataloader(cfgs, datype="train"):

    dataset_cfgs = OmegaConf.load(cfgs.dataset_cfg_path)
    print(f"Extracting data from {dataset_cfgs.target}")
    Dataset = eval(dataset_cfgs.target)
    dataset = Dataset(dataset_cfgs.params, datype = datype)

    return data.DataLoader(dataset=dataset, batch_size=cfgs.batch_size, shuffle=cfgs.shuffle, num_workers=cfgs.num_workers, drop_last=True)

