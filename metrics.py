import lpips
import os,glob
from os.path import join as ospj

def calc_fid(fake_dir, real_dir, batch_size=1, gpu='0'):

    print(f"evaluating FID score between '{fake_dir}' and '{real_dir}'")

    os.system(f"python -m pytorch_fid {fake_dir} {real_dir} --batch-size {batch_size} --device cuda:{gpu}")


def calc_lpips(fake_dir, real_dir):

    print(f"evaluating LPIPS score between '{fake_dir}' and '{real_dir}'")

    loss_fn = lpips.LPIPS(net='alex').cuda()

    fake_paths = sorted(glob.glob(ospj(fake_dir, "*")))
    real_paths = sorted(glob.glob(ospj(real_dir, "*")))

    dists = []
    for fake_path, real_path in zip(fake_paths, real_paths):

        fake_img = lpips.im2tensor(lpips.load_image(fake_path)).cuda() # RGB image from [-1,1]
        real_img = lpips.im2tensor(lpips.load_image(real_path)).cuda()
    
        dist = loss_fn.forward(fake_img, real_img)
        dists.append(dist)
    
    print(f"lpips score: {sum(dists)/len(dists)}")

