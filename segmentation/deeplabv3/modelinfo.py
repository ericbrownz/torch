from torchinfo import summary
from src import deeplabv3_resnet50

model = deeplabv3_resnet50(aux=True, num_classes=21)
batch_size = 16
summary(model, input_size=(16, 3, 480, 480))