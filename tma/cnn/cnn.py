import mnist
from tma.cnn.conv3x3 import Conv3x3
from tma.cnn.maxpool2 import MaxPool2

train_images = mnist.train_images()
train_labels = mnist.train_labels()
conv = Conv3x3(8)
pool = MaxPool2()

outputc = conv.forward(train_images[0])
print(outputc.shape)
outputp = pool.forward(outputc)
print(outputp.shape)
