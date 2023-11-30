import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


class ParseqPredictor(nn.Module):

    def __init__(self, ckpt_path=None, freeze=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parseq = torch.hub.load('./src/parseq', 'parseq', source='local').eval()
        self.parseq.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.parseq_transform = transforms.Compose([
            transforms.Resize(self.parseq.hparams.img_size, transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(0.5, 0.5)
        ])

        if freeze:
            self.freeze()

    def freeze(self):
        for param in self.parseq.parameters():
            param.requires_grad_(False) 

    def forward(self, x):
        
        x = torch.cat([self.parseq_transform(t[None]) for t in x])
        logits = self.parseq(x.to(next(self.parameters()).device))

        return logits

    def img2txt(self, x):

        pred = self(x)
        label, confidence = self.parseq.tokenizer.decode(pred)
        return label

    
    def calc_loss(self, x, label):

        preds = self(x)  # (B, l, C) l=26, C=95
        gt_ids = self.parseq.tokenizer.encode(label).to(preds.device) # (B, l_trun)

        losses = []
        for pred, gt_id in zip(preds, gt_ids):

            eos_id = (gt_id == 0).nonzero().item()
            gt_id = gt_id[1: eos_id]
            pred = pred[:eos_id-1, :]

            ce_loss = nn.functional.cross_entropy(pred.permute(1, 0)[None], gt_id[None])
            ce_loss = torch.clamp(ce_loss, max = 1.0)
            losses.append(ce_loss[None])

        loss = torch.cat(losses)

        return loss