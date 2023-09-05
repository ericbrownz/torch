from torchinfo import summary
from src import fcn_resnet50

model = fcn_resnet50(aux=True, num_classes=21)
batch_size = 16
summary(model, input_size=(16, 3, 480, 480))