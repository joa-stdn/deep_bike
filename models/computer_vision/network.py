import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

from constants import *

class Network:
    
    def __init__(self,classes=2,inner_nodes=256,name = 'nn1',learning_rate = 0.0001, regularization = 0.001):#,train_batches,validation_batches,test_batches):
        self.name = name
        
        # Create the base model from the pre-trained model MobileNet V2
        base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        
        X=base_model.output
        X=GlobalAveragePooling2D()(X)
        X=Dense(inner_nodes,activation='relu', kernel_regularizer=regularizers.l2(regularization))(X)
        preds=Dense(classes,activation='softmax')(X)
        
        self.model=Model(inputs=base_model.input,outputs=preds)
        
        self.model.compile(optimizer=Adam(lr=learning_rate),
                           loss=CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        
        
    
    def train(self,train_generator,validation_generator,epochs,verbose = 1):
        step_size_train=train_generator.n//train_generator.batch_size
        step_size_val=validation_generator.n//validation_generator.batch_size
        
        history = self.model.fit(train_generator,
                                 steps_per_epoch = step_size_train,
                                 validation_data = validation_generator, 
                                 validation_steps = step_size_val,
                                 epochs=epochs, verbose=verbose)   
        return history     
    
    def predict(self,pred_generator,labels,verbose = 1):
        step_size_pred=pred_generator.n//pred_generator.batch_size
        pred_generator.reset()
        pred=self.model.predict(pred_generator,
                                steps=step_size_pred,
                                verbose=verbose)
        
        predicted_class_indices=np.argmax(pred,axis=1)
        labels = dict((v,k) for k,v in (labels).items())
        predictions = [labels[k] for k in predicted_class_indices]
        filenames = pred_generator.filenames
        return [predictions, filenames]
    
    def load(self, name = None):
        if name == None:name = self.name
        self.model.load_weights('models/weights of '+name+'.nn')
    
    def save(self, name = None):
        if name == None:name = self.name
        self.model.save_weights('models/weights of '+name+'.nn')
    
    def plot_learning_curves(self,history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()    