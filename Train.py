import time
import chainer.computational_graph
import chainer.serializers
from chainer import  Variable
import numpy as np

# batch learner 
class Train:
    def __init__(self, model, optimizer, xp=np):
        self.model = model
        self.optimizer = optimizer
        self.xp = xp
        self.ratio = 0.8

    def loop_train(self, X_t, Y_t, batch_size, max_epoch, prefix, cur_epoch=0):
        '''
        train multiple times
        '''
        total_start = time.time()
        for epoch in range(cur_epoch, max_epoch):
            epoch_str = epoch+1
            print('Epoch %d' % (epoch_str))
            start = time.time()
            loss, accuracy = self.train(X_t, Y_t, batch_size)
            ##################################################################
            # save network graph only at first epoch
            if epoch == 1:
                with open('%s.dot'%prefix, 'wb') as f:
                    g = chainer.computational_graph.build_computational_graph((self.model.loss,), remove_split=True)
                    f.write(g.dump())
                    f.flush()
                    f.close()
            ##################################################################
            # dump model and optimizer after each 10 cycle. 
            # epoch+1 during dump to make it more readable
            if epoch_str % 5 == 0:
                print('Dump model of epoch %d to %s' % (epoch_str, '%s-%s.model' % (prefix, str(epoch_str).zfill(4))))
                chainer.serializers.save_npz('%s-%s.model' % (prefix, str(epoch_str).zfill(4)), self.model)
                print('Dump optimizer of epoch %d to %s' % (epoch_str, '%s-%s.state' % (prefix, str(epoch_str).zfill(4))))
                chainer.serializers.save_npz('%s-%s.state' % (prefix, str(epoch_str).zfill(4)), self.optimizer)
            ##################################################################
            # print out the train loss accuracy and validation loss and accuracy
            print('train_mean_loss=%f, acurracy=%f' % (loss, accuracy))
            print('val_mean_loss=%f, accuracy=%f' % self.evaluate(X_t, Y_t, batch_size))
            print('Time elapsed in epoch %s: %f seconds.'%(str(epoch_str).zfill(4), time.time()-start))
            ##################################################################
        print('Time elapsed totally  %f seconds.'%(time.time()-total_start))
    
    def train(self, X_t, Y_t, batch_size):
        '''
            batch train
        '''
        num = int(len(X_t)*self.ratio)
        sum_loss, sum_accuracy = 0.0, 0.0
        for i in range(0, num, batch_size):
            x_train = X_t[i:(i+batch_size)]
            y_train = Y_t[i:(i + batch_size)]
            x = Variable(self.xp.asarray(x_train))
            t = Variable(self.xp.asarray(y_train))
            self.optimizer.update(self.model, x, t)
            sum_loss += float(self.model.loss.data) * batch_size
            sum_accuracy += float(self.model.accuracy.data) * batch_size
        mean_loss, mean_accuracy =  sum_loss/num, sum_accuracy/num
        return mean_loss, mean_accuracy
        
    def evaluate(self, X_t, Y_t, batch_size):
        '''
            batch evaluator
        '''
        num = int(len(X_t)*(1-self.ratio))
        sum_loss, sum_accuracy = 0.0, 0.0
        for i in range(0, num, batch_size):
            x_val = X_t[i:(i+batch_size)]
            y_val = Y_t[i:(i + batch_size)]
            x = Variable(self.xp.asarray(x_val))
            t = Variable(self.xp.asarray(y_val))
            loss = self.model(x, t, train=False)
            sum_loss += loss.data * batch_size
            sum_accuracy += self.model.accuracy.data * batch_size
        mean_loss, mean_accuracy = sum_loss / num, sum_accuracy / num
        return mean_loss, mean_accuracy
