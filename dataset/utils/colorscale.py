import colorlover as cl
import random
from PIL import Image, ImageDraw, ImageFont


class ColorScale():

    def __init__(self):

        self.color_pairs = []
        for s in range(3,8):
            color_seqs = cl.scales[str(s)]['seq']
            for k in color_seqs:
                color_seq = color_seqs[k]
                for i in range(len(color_seq)-1):
                    self.color_pairs.append((color_seq[i], color_seq[i+1]))

        self.color_pairs.append(('rgb(0,0,0)', 'rgb(255, 255, 255)'))
        self.color_pairs.append(('rgb(255,255,255)', 'rgb(0,0,0)'))
        self.length = len(self.color_pairs)
    
    def get_pairs(self):

        return random.choice(self.color_pairs)
    


# cs = ColorScale()
# p = cs.get_pairs()
# ref = Image.new('RGB', (512,512), color = p[0])
# ref.save("0.png")
# ref = Image.new('RGB', (512,512), color = p[1])
# ref.save("1.png")




