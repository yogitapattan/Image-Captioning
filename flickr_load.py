import os
import pickle
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from os import listdir
import tensorflow.contrib.keras as keras
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.backend import manual_variable_initialization
manual_variable_initialization(True)

class Imagecap:

    def VGGModel(self):
        image_model = VGG16(include_top = True, weights='imagenet')
        transfer_layer = image_model.get_layer('fc2')
        image_model_transfer = Model(inputs =image_model.input, outputs = transfer_layer.output)
        return image_model_transfer,transfer_layer.output


    def get_imgsignature(self,image_model_transfer,Img,num):
        trans_values = []
        for _ in range(num):
            trans_values.append(image_model_transfer.predict(Img[_]))
        return trans_values

    def invert_dict(self,d):
        return dict([ (v, k) for k, v in d.items( ) ])


    def decoder(self, state_size,num_words,embedding_size,transfer_layer_output):
        transfer_values_size = K.int_shape(transfer_layer_output)[1]
        transfer_values_input = Input(shape=(transfer_values_size,),name='transfer_values_input')
        decoder_transfer_map = Dense(state_size,activation='tanh',name='decoder_transfer_map')
        decoder_input = Input(shape=(None, ), name='decoder_input')
        decoder_embedding = Embedding(input_dim=num_words,output_dim=embedding_size,name='decoder_embedding')   
        dropout_1 = Dropout(0.2)
        decoder_gru1 = GRU(state_size, name='decoder_gru1',return_sequences=True)
        decoder_dense = Dense(num_words,activation='linear',name='decoder_output')
        #self.decoder_model = self.decoder(self.state_size,self.num_words,self.embedding_size,self.num_words,self.transfer_layer_output)



        initial_state = decoder_transfer_map(transfer_values_input)
        initial_state = dropout_1(initial_state)
        net = decoder_input
        net = decoder_embedding(net)
        net = decoder_gru1(net, initial_state=initial_state)
        decoder_output = decoder_dense(net)
        decoder_model = Model(inputs=[transfer_values_input, decoder_input],outputs=[decoder_output])

        return decoder_model
    

    def sparse_cross_entropy(self,y_true, y_pred):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
        loss_mean = tf.reduce_mean(loss)
        return loss_mean

    def generate_caption(self,image_model_transfer,image_path, unique_words_dict, max_tokens=30):
        token_end = unique_words_dict["eeee"]
        image = load_img(image_path, target_size=(224, 224))
        a = image
        
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        transfer_values = image_model_transfer.predict(image)
        
        shape = (1, max_tokens)
        
        decoder_input_data = np.zeros(shape=shape, dtype=np.int)
        token_int = unique_words_dict["ssss"]
        output_text = ''
        count_tokens = 0
        
        while token_int != token_end and count_tokens < max_tokens:
            
            decoder_input_data[0, count_tokens] = token_int
            
            x_data = \
            {
                'transfer_values_input': transfer_values,
                'decoder_input': decoder_input_data
            }
            decoder_output = self.decoder_model.predict(x_data)
            
            token_onehot = decoder_output[0, count_tokens, :]

            # Convert to an integer-token.
            token_int = np.argmax(token_onehot)
            sampled_word = self.inttoword[token_int]
            
            # Append the word to the output-text.
            output_text += " " + sampled_word

            # Increment the token-counter.
            count_tokens += 1
        output_tokens = decoder_input_data[0]
        #plt.imshow(a)
        #plt.show()
        print("Image Path" + image_path)
        print("Predicted caption:")
        print(output_text)
        return output_text
        print()
            
    def load_dictionary(self, directory):
        pickle_in = open(directory,'rb')
        wordtoint = pickle.load(pickle_in)
        pickle_in.close()
        return wordtoint

    def __init__(self):
        self.wordtoint = self.load_dictionary('int2word.pickle')
        self.inttoword = self.invert_dict(self.wordtoint)
        print('Step 1')
        self.model, self.transfer_layer_output = self.VGGModel()
        self.state_size = 512
        self.embedding_size = 128
        self.num_words = 4659
        self.batch_size = 32
        self.num_images = 0
        print('Step 2')
        self.decoder_model = self.decoder(self.state_size,self.num_words,self.embedding_size,self.transfer_layer_output)
        print('Step 3')
        self.decoder_model.load_weights("gru_1_150_epoch_plus.keras")
        mode = self.model
        
    
print('here goes')
img_cap1 = Imagecap()

aa= 't8.jpg'
bb = img_cap1.generate_caption(img_cap1.model,aa,img_cap1.wordtoint,max_tokens=30)

