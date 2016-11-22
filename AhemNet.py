import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F

nb_filters = 32


class AhemNet(chainer.Chain):
    """
    AhemNet
    - It takes (64, 64, 1) sized image as imput
    """

    def __init__(self, num_classes=2):
        super(AhemNet, self).__init__(
            conv1=L.Convolution2D(None, nb_filters, ksize=3, stride=1),
            conv2=L.Convolution2D(None, nb_filters, ksize=3, stride=1),

            conv3=L.Convolution2D(None, nb_filters, ksize=3, stride=1, pad=1),
            conv4=L.Convolution2D(None, nb_filters, ksize=3, stride=1),

            fc5=L.Linear(None, 128),
            fc6=L.Linear(128, num_classes)
        )

    def __call__(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, train=train, ratio=0.25)

        h = F.relu(self.conv3(x))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, train=train, ratio=0.25)

        #h = F.flatten(h)

        h = F.relu(self.fc5(h))
        h = F.dropout(h, train=train, ratio=0.5)

        h = self.fc6(h)
        return h
