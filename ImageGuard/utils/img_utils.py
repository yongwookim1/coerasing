from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F

from .ixc_utils import HD_transform

class Resize_with_pad:
    def __init__(self, w=490, h=490):
        self.w = w
        self.h = h

    def __call__(self, image):
        w_1, h_1 = image.size
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1
        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w], interpolation=InterpolationMode.BICUBIC)

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w], interpolation=InterpolationMode.BICUBIC)

        else:
            return F.resize(image, [self.h, self.w], interpolation=InterpolationMode.BICUBIC)

class ImageProcessor:

    def __init__(self, image_size=224):
        self.resizepad = Resize_with_pad(image_size, image_size)
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose([
            # transforms.Resize((image_size, image_size),
                            #   interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, itemname):
        try:
            if isinstance(itemname, Image.Image):
                item = itemname.convert('RGB')
            else:
                item = Image.open(itemname).convert('RGB')
            item = self.resizepad(item)
        except Exception as e:
            print(e, flush=True)
            print('error img', itemname, flush=True)
            exit()
        return self.transform(item)

class ImageProcessorHD:

    def __init__(self, image_size=224, hd_num=-1):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)
        self.hd_num = hd_num

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        item = Image.open(item).convert('RGB')
        return self.transform(HD_transform(item, hd_num=self.hd_num))

    
def get_internlm_processor():
    return ImageProcessor(image_size=490)


processor_dict = {
    'Internlm': get_internlm_processor,
}

def get_image_processor(model_name):
    return processor_dict[model_name]()