# %%
import os,glob,sys
import zipfile
import json
import requests
from tqdm import tqdm
from os.path import join as ospj

# %%
data_root = "/data/zhaoym/TextDesign-XL/LAION-OCR"
image_root = ospj(data_root, "image")
anno_root = ospj(data_root, "annotation")
cache_root = ospj(data_root, "cache")
os.makedirs(cache_root, exist_ok=True)

# %%
url_txt = "/data/zhaoym/TextDesign-XL/LAION-OCR/mario_laion_image_url/mario-laion-index-url.txt"
with open(url_txt, 'r') as fp:
    res = fp.readlines()

url_lst = []
for r in res:
    idx, url = r.split(" ")
    url = url[:-1]
    ex_idx, in_idx = idx.split("_")
    if int(ex_idx) >= 50000: continue
    url_lst.append({"ex_idx": ex_idx, "in_idx": in_idx, "url": url})
len(url_lst)

# %%
url_lst.sort(key = lambda x: int(x["in_idx"]))
# url_lst

# # %%
# url_txt_path = ospj(data_root, "urls.txt")
# os.system(f"rm {url_txt_path}")
# urls = []
# for item in tqdm(url_lst[1500000:3000000]):
#     ex_idx = item["ex_idx"]
#     in_idx = item["in_idx"]
#     url = item["url"]
#     urls.append(url+"\n")

# with open(url_txt_path, "w") as fp:
#     fp.writelines(urls)

# # %%
# print(f"img2dataset --url_list={url_txt_path} --output_folder={cache_root} --thread_count=64  --resize_mode=no")

# %%
# pointer = 0
# total = 0
# ex_dirs = sorted(glob.glob(ospj(cache_root, "?????")))
# for ex_dir in tqdm(ex_dirs):
#     img_paths = sorted(glob.glob(ospj(ex_dir, "*.jpg")))
#     info_paths = sorted(glob.glob(ospj(ex_dir, "*.json")))
#     total += len(info_paths)
#     for img_path in img_paths:
#         name = img_path.split(os.sep)[-1].split(".")[0]
#         with open(ospj(ex_dir, f"{name}.json"), "rb") as fp:
#             info = json.load(fp)

#         while info["url"] != url_lst[pointer]["url"]:
#             pointer += 1
#         if pointer >= len(url_lst):
#             print("pointer error")
#         ex_idx = url_lst[pointer]["ex_idx"]
#         in_idx = url_lst[pointer]["in_idx"]
#         os.makedirs(ospj(image_root, ex_idx), exist_ok=True)
#         os.rename(img_path, ospj(image_root, ex_idx, f"{in_idx}.jpg"))

# print(f"Total num: {total}")

# %%
img_dirs = sorted(glob.glob(ospj(image_root, "?????")))
div = int(len(img_dirs) * 0.95)
train_total = 0
val_total = 0
for i, img_dir in enumerate(tqdm(img_dirs)):
    stype = "train" if i < div else "val" 
    img_paths = sorted(glob.glob(ospj(img_dir, "*.jpg")))
    for img_path in img_paths:
        if stype == "train":
            train_total += 1
        else:
            val_total += 1
        ex_idx = img_dir.split(os.sep)[-1]
        in_idx = img_path.split(os.sep)[-1].split(".")[0]
        anno_dir = ospj(anno_root, ex_idx, in_idx)
        target_dir = ospj(data_root, stype, in_idx)
        target_img_path = ospj(target_dir, 'image.jpg')
        if not os.path.exists(anno_dir):
            continue
        if os.path.exists(target_img_path):
            continue
        os.system(f"cp -r {anno_dir} {target_dir}")
        os.system(f"cp {img_path} {target_img_path}")


print(f"div {train_total} train samples and {val_total} val samples")
        



