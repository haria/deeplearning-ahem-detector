from chainer import Chain
import chainer.functions as F

class Classifier(Chain):
    '''
    Classifier for Ahem Network, compute loss values and evaluate the accuracy of the predictions
    '''
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t, train=True):
        y = self.predictor(x, train)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss

    def predict(self, x, train=False):
        y = self.predictor(x, train)
        return y
