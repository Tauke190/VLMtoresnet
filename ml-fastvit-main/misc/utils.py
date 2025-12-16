

from torchvision.utils import save_image

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())



def dump_images(x, name="temp.png"):
    x  = normalize(x)
    save_image(x, name)