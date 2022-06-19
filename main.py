import base64
from fastapi import FastAPI
import time
from fastapi.middleware.cors import CORSMiddleware
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

def extract_features(filename, vgg_16_model):
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1],
                          image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = vgg_16_model.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
    for (word, index) in list(tokenizer.word_index.items()):
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(
    model,
    tokenizer,
    photo,
    max_length,
    ):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

def predict_caption(filename, vgg_16_model):
    photo = extract_features(filename, vgg_16_model)

    # generate description

    description = generate_desc(model, tokenizer, photo, max_length)
    # Remove startseq and endseq

    query = description
    stopwords = ['startseq', 'endseq']
    querywords = query.split()

    resultwords = [word for word in querywords if word.lower()
                not in stopwords]
    result = ' '.join(resultwords)
    return result


# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('model_18.h5')
# load vgg_16 model
vgg_16_model = VGG16()
# edit model
vgg_16_model = Model(inputs=vgg_16_model.inputs, outputs=vgg_16_model.layers[-2].output)



# Create application
app = FastAPI(title='Speech Server')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/caption/get")
def caption(file: str):
    imgstr = file
    imgdata = base64.b64decode(imgstr + "========")
    filename = 'getcaption.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    caption = predict_caption(filename, vgg_16_model)
    print(caption)
    return {"caption": caption}
