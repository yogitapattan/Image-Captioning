import os 
import tensorflow as tf
import pickle
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


def load_imagesdict(directory):
    images = dict()
    count = 0 
    for name in listdir(directory):
        if(count%200==0):
           print(count)
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get image id
        image_id = name.split('.')[0]
        images[image_id] = image
        count = count + 1
    return images
  


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text



def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # store the first description for each image
        if image_id not in mapping:
            mapping[image_id] = image_desc
    return mapping


def loadcaptionsdict(filename):
    doc = load_doc(filename)
    descriptions = load_descriptions(doc)
    return descriptions



def getimgandcaptionlist(images,descriptions):
    imgids = list(images.keys())
    Images = []
    captions = []
    for _ in range(len(imgids)):
        Images.append(images[imgids[_]])
        captions.append(descriptions[imgids[_]])
    return Images,captions


def mark_captions(captions_list):
    mark_start= 'ssss '
    mark_end = ' eeee'
    captions_marked = [mark_start + caption + mark_end for caption in captions_list]
    return captions_marked




def sentences_to_listt(caps):
    listowords = []
    for _ in caps:
        w = _.split()
        if "." in w:
            w.remove(".")
        listowords.append(w)
    return listowords



def get_unique_wordsdict(captions_marked_list):
    unique_words = []
    for i in range(len(captions_marked_list)):
        for j in range(len(captions_marked_list[i])):
            unique_words.append(captions_marked_list[i][j])
    unique_words = list(set(unique_words))
    inttoword = dict(list(enumerate(unique_words)))
    return inttoword

def invert_dict(d):
    return dict([ (v, k) for k, v in d.items( ) ])

def word2int(caps,unique_words_dict):
    for i in range(len(caps)):
        for j in range(len(caps[i])):
            caps[i][j] = unique_words_dict[caps[i][j]]
    return caps


def VGGModel():
    image_model = VGG16(include_top = True, weights='imagenet')
    transfer_layer = image_model.get_layer('fc2')
    image_model_transfer = Model(inputs =image_model.input, outputs = transfer_layer.output)
    return image_model_transfer,transfer_layer.output

def get_imgsignature(image_model_transfer,Img,num):
    trans_values = []
    for _ in range(num):
        if(_%1000==0):
          print(_)
        trans_values.append(image_model_transfer.predict(Img[_]))
    return trans_values


def decoder(statesize,embeddingsize,maxnumwords,transfer_layer_output):
    transfer_values_size = K.int_shape(transfer_layer_output)[1]
    transfer_values_input = Input(shape=(transfer_values_size,),name='transfer_values_input')
    decoder_transfer_map = Dense(state_size,activation='tanh',name='decoder_transfer_map')
    decoder_input = Input(shape=(None, ), name='decoder_input')
    decoder_embedding = Embedding(input_dim=num_words,output_dim=embedding_size,name='decoder_embedding')   
    dropout_1 = Dropout(0.2)
    decoder_gru1 = GRU(state_size, name='decoder_gru1',return_sequences=True)
    decoder_dense = Dense(num_words,activation='linear',name='decoder_output')
    


    initial_state = decoder_transfer_map(transfer_values_input)
    initial_state = dropout_1(initial_state)
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net, initial_state=initial_state)
    decoder_output = decoder_dense(net)
    decoder_model = Model(inputs=[transfer_values_input, decoder_input],outputs=[decoder_output])

    return decoder_model

def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean


def batch_generator(batch_size):
    """
    Generator function for creating random batches of training-data.
    
    Note that it selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """

    # Infinite loop.
    while True:
        # Get a list of random indices for images in the training-set.
        idx = np.random.randint(len(Images),size=batch_size)
        
        # Get the pre-computed transfer-values for those images.
        # These are the outputs of the pre-trained image-model.
        transf_values = np.array([transfer_values[_] for _ in idx])

        # For each of the randomly chosen images there are
        # at least 5 captions describing the contents of the image.
        # Select one of those captions at random and get the
        # associated sequence of integer-tokens.
        tokens = [caps_markedwords[_] for _ in idx]

        # Count the number of tokens in all these token-sequences.
        num_tokens = [len(t) for t in tokens]
        
        # Max number of tokens.
        max_tokens = np.max(num_tokens)
        # Pad all the other token-sequences with zeros
        # so they all have the same length and can be
        # input to the neural network as a numpy array.
        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')
        
        # Further prepare the token-sequences.
        # The decoder-part of the neural network
        # will try to map the token-sequences to
        # themselves shifted one time-step.
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        # Dict for the input-data. Because we have
        # several inputs, we use a named dict to
        # ensure that the data is assigned correctly.
        x_data = \
        {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transf_values
        }


        # Dict for the output-data.
        y_data = \
        {
            'decoder_output': decoder_output_data
        }
        
        yield (x_data, y_data)



def generate_caption(image_model_transfer,image_path, unique_words_dict, max_tokens=30):
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
        decoder_output = decoder_model.predict(x_data)
        
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)
        sampled_word = inttoword[token_int]
        
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
	

directory = 'Flicker8k_Dataset'
filename = 'Flickr8k.token.txt'
state_size = 512
embedding_size = 128
num_words = 4659
batch_size = 32
num_images = 0
captions_marked = []
caps_markedwords = []
captions_marked = []
caps_markedwords = []
print('loading dictionaries')
'''
pickle_out = open("word2int.pickle","rb")
wordtoint = pickle.load(pickle_out)
pickle_out.close()
'''
pickle_out = open("int2word.pickle","rb")
wordtoint = pickle.load( pickle_out)
pickle_out.close()

inttoword = invert_dict(wordtoint)

model,transfer_layer_output = VGGModel()
decoder_model = decoder(state_size,embedding_size,num_words,transfer_layer_output)
optimizer = RMSprop(lr=1e-3)
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
decoder_model.compile(optimizer=optimizer,loss=sparse_cross_entropy,target_tensors=[decoder_target])
print(len(inttoword))
print(len(wordtoint))
print(wordtoint.keys())
path_checkpoint = 'gru_1_150_epoch_plus.keras'

#path_checkpoint = '/content/drive/My Drive/data/gru_1_150_epoch_plus.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,verbose=1,save_weights_only=True)
callback_tensorboard = TensorBoard(log_dir='./22_logs/',histogram_freq=0,write_graph=False)
callbacks = [callback_checkpoint, callback_tensorboard]
generator = batch_generator(batch_size)
print('loading checkpoint')

decoder_model.load_weights(path_checkpoint)

aa ='t8.jpg'
generate_caption(model,aa,wordtoint,max_tokens=30)



