import chainer.serializers
from chainer import    Variable
import chainer.functions as F
import chainer.cuda as cuda
import numpy as np
import os, sys, time
import requests
import argparse
import skimage.io as io

from Classifier import Classifier
from AhemNet import AhemNet


def load_image(filename):
    img = io.imread(filename)
    img = img.transpose((2, 0, 1))
    img = img[:3, :, :]
    return img


class Predictor:
    def __init__(self, model, model_path, xp = np):
        self.model = model
        if cuda.available:
            print ("load model to GPU")
            self.model.to_gpu()
        print ("Predictor model from: %s loading."%model_path)
        chainer.serializers.load_npz(model_path, self.model)
        self.xp = xp
        self.train=False

    def predict(self, x, top_k=1):
        x = Variable(self.xp.asarray(x, 'float32')/255.0)
        t = self.model.predict(x, train=self.train)
        prob = chainer.cuda.to_cpu(F.softmax(t).data)
        pred = np.argsort(prob)[0][::-1]
        index = pred[:top_k]

        score = [prob[0][idx] for idx in index]
        return (index, score)
        
    def predict_image(self, image_path, top_k=1):
        img = load_image(image_path)
        batch_data= [img]
        return self.predict(batch_data, top_k=top_k)
         
if __name__=="__main__":
    ##########################################################
    # parse argument input from user
    parser = argparse.ArgumentParser(description='Chainer AhemNet')
    parser.add_argument('--model', default='./AhemDetector-0020.model', help='Initialize the model from given file')
    parser.add_argument('--gpu', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--numclasses', default=2, type=int, help='ahem or not')
    parser.add_argument('--image', help='test image path')
    args = parser.parse_args()
    ##########################################################
    # cupy or numpy
    if cuda.available and args.gpu >=0 :
        print ("CUDA is enabled, use device %d."%args.gpu)
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np

    model = Classifier(AhemNet(num_classes=args.numclasses))
        
    predictor = Predictor(model, args.model, xp)
    if os.path.exists(args.image) and os.path.isfile(args.image):
        result = predictor.predict_image(args.image)
        print args.image, result
    else:
        image_path= args.image
        import glob
        print image_path
        images = glob.glob(os.path.join(image_path, "*.png"))
        for image in images:
            result = predictor.predict_image(image)
            print image, result
