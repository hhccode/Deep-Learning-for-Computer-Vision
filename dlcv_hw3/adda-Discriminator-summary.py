from models import ADDA
from torchsummary import summary

model = ADDA(batch_size=64, device="cpu")
summary(model.discriminator, input_size=(128*5*5,))
