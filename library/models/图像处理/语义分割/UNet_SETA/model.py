from models.图像处理.语义分割.UNet_SETA.Unet_ASE_V3 import UNetWithAttention


def model(num_classes=3):
    model = UNetWithAttention(num_classes=num_classes)
    return model
