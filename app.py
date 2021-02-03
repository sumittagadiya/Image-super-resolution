import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image



from flask import Flask,flash, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

########################### PATH ###########################################
# model architacture path
app.config['MODEL_JSON_PATH'] = 'saved_model/edsr_model.json'
# trained model weights
app.config['MODEL_PATH_BICUBIC'] = 'saved_model/edsr_350000.h5'
app.config['MODEL_PATH_UNKNOWN'] = 'saved_model/edsr_unknown_x4.h5'
app.config['LETTER_SET'] = list(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
# Uploaded image folder path
app.config['UPLOAD_FOLDER'] = 'static/uploaded'
# predicted image folder path
app.config['PRED_FOLDER ']= 'static/predicted'

############################## Load pre-trained model #########################
json_file = open(app.config['MODEL_JSON_PATH'], 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(app.config['MODEL_PATH_BICUBIC'])
# load unknown_x4 weights
model1 = tf.keras.models.model_from_json(loaded_model_json)
model1.load_weights(app.config['MODEL_PATH_UNKNOWN'])
print('Both Model Loaded succesfully')

#################### generate random file name #######################
def generate_random_name(filename):
    """ Generate a random name for an uploaded file. """
    ext = filename.split('.')[-1]
    rns = [random.randint(0, len(app.config['LETTER_SET']) - 1) for _ in range(3)]
    chars = ''.join([app.config['LETTER_SET'][rn] for rn in rns])

    new_name = "{new_fn}.{ext}".format(new_fn=chars, ext=ext)
    new_name = secure_filename(new_name)

    return new_name

#################### open image #################################
def load_image(path):
    return np.array(Image.open(path))

#################### Get predicted image ###########################
def predict(lr_path,model):
    '''' Get super resolution image from model'''
    lr_image = load_image(lr_path)
    # expand dims of lr image for prediction
    if len(lr_image.shape) == 3 :
        lr_image = tf.expand_dims(lr_image,axis=0)
     
    # convert dtype to float32
    lr_img = tf.cast(lr_image, tf.float32)

    # predict image
    sr_img = model(lr_img)
    sr_img = tf.clip_by_value(sr_img, 0, 255)
    sr_img = tf.round(sr_img)
    sr_img = tf.cast(sr_img, tf.uint8)
    return sr_img[0]

################ main route ########################################

@app.route('/')

def upload():
    return render_template('index.html')




@app.route('/predicted',methods=['POST'])
def predicted():
    # if any other extension then img file then error raise
    if 'image' not in request.files :
        flash('No file was uploaded.')
        return redirect(request.url)

    # Get the file from post request
    image_file = request.files['image']
    
    # if not image uploaded then redirect
    if image_file.filename == '' :
        return redirect(request.url)
    
    # if image is uploaded then do further process
    if image_file :
        ############### if model = edsr_unknown #######################
        if str(request.form.get('models')) == 'edsr_unknown' :
            print('***** Using EDSR Unknown_x4 model weights ******')
            # Save the file to ./uploads
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],image_file.filename)
            # save uploaded image
            image_file.save(file_path)

            # Make prediction
            preds = predict(file_path, model1)
            # convert array to img
            pred_img = tf.keras.preprocessing.image.array_to_img(preds)
            # generate random image file name for predicted image
            pred_filename = generate_random_name(image_file.filename)
            # predicted image absolute path
            pred_filepath = os.path.join(app.config['PRED_FOLDER '],pred_filename)
            # save predicted image
            pred_img.save(pred_filepath)


            return render_template('predicted.html',
            uploaded_image=image_file.filename,pred_image=pred_filename)

        #################### if model = bicubic_x4 #####################

        print('************ Using bicubic_x4 model weights **********')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],image_file.filename)
        # save uploaded image
        image_file.save(file_path)
        # Make prediction
        preds = predict(file_path, model)
        # convert array to img
        pred_img = tf.keras.preprocessing.image.array_to_img(preds)
        # generate random image file name for predicted image
        pred_filename = generate_random_name(image_file.filename)
        # predicted image absolute path
        pred_filepath = os.path.join(app.config['PRED_FOLDER '],pred_filename)
        # save predicted image
        pred_img.save(pred_filepath)


        return render_template('predicted.html', 
        uploaded_image=image_file.filename, pred_image=pred_filename)


if __name__ == '__main__':
    
    app.run(debug=True)
