import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar


def find_threshold(y_true, y_pred_proba):
    y_true = y_true.ravel().astype(np.int32)
    y_pred_proba = y_pred_proba.ravel()
    search = minimize_scalar(lambda x: -f1_score(y_true, (y_pred_proba >= x).astype(np.int32)),
                             method='bounded',
                             bounds=(0.0, 1.0))
    return search.x


class F1_EarlyStopping(Callback):
    def __init__(self, train_X, train_y, val_X, val_y, batch_size):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.batch_size = batch_size
        self.best_score = -1.0
        self.best_weights = []
        super(F1_EarlyStopping, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        print('Epoch {0} train finished. Checking classification quality.'.format(epoch + 1))
        train_pred_proba = self.model.predict(self.train_X, batch_size=self.batch_size)
        val_pred_proba = self.model.predict(self.val_X, batch_size=self.batch_size)
        threshold = find_threshold(self.train_y, train_pred_proba)
        val_pred = (val_pred_proba.ravel() >= threshold).astype(np.int32)
        f1 = f1_score(self.val_y.ravel().astype(np.int32),
                      val_pred)
        print('Val F1: {0}'.format(f1))
        if f1 >= self.best_score:
            print('Updated model')
            self.best_score = f1
            self.best_weights = self.model.get_weights()
        else:
            print('Finished training. Returned to best state.')
            self.model.set_weights(self.best_weights)
            self.model.stop_training = True
