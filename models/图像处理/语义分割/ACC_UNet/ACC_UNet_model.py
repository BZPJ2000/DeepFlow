from models.图像处理.语义分割.ACC_UNet.ACC_UNet import ACC_UNet



def model(n_classes=3,num_classes=4):
    model = ACC_UNet(n_classes=n_classes,n_channels=num_classes)
    return model


print(model())