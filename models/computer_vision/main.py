from datacreator import *
from network import *
from constants import *

train_gen,val_gen = DataCreator().train_generators()
pred_gen = DataCreator().predict_generator()

net = Network(classes = CLASSES,inner_nodes = INNER_NODES, name = 'my predictor',learning_rate=0.00003, regularization = 0.2)

#net.load()

hist = net.train(train_gen,val_gen,2,verbose = 1)

net.save()

pred,filenames = net.predict(pred_gen, train_gen.class_indices,verbose = 1)
for i in range(len(pred)):
    print(filenames[i]+' belongs to '+pred[i])

# Plots loss and accuracy histories on the train and dev set
net.plot_learning_curves(hist)