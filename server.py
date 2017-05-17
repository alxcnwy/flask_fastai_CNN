##############################
### ML IMAGE DETECTION SERVER
##############################

#############
#### IMPORTS
#############

# Flask
from flask import Flask, render_template, make_response, request, Response, redirect, send_from_directory, jsonify

# Flask Misc
from flask import request
import json

# Misc
import datetime, time
import os, sys
import shutil

print '### start server ', str(datetime.datetime.now())

current_dir = os.getcwd()
uploadsfolder = current_dir + "/static/_uploads/unknown/"
print "\n### image upload folder: " + uploadsfolder

 # $$$ 
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir+'/static/data/redux'
print "\n### data folder: " + DATA_HOME_DIR

# Allow relative imports for keras wrapper modules in /utils/
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
                
#import keras wrapper modules
from utils import *
from vgg16 import Vgg16

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# make theano multi-threaded
import theano
theano.config.openmp = True

# print options
np.set_printoptions(precision=5, suppress=True)

print "\n### initializing model: "
# instantiate model with pre-trained weights
vgg = Vgg16()

# images to score at once
batch_size=16

# setup path with images to be scored
# due to the way the model works, need all images to be inside an "unknown" folder
test_path = uploadsfolder.replace("unknown/","")
print "\n### test path: " + test_path





##################
#### FLASK SETUP
##################
app = Flask(__name__)

# app config
app.config['DEBUG'] = True

app.static_folder = 'static/'
app.static_url_path = 'static/'

# 
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

###################
### FLASK - ROUTES
###################

# Return dash page
@app.route('/')
def index():
    return render_template('Upload.html')


# POST IMAGE 
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():

    if request.method =='POST':
        submitted_file = request.files['data_file']
        if submitted_file:
            # get extension
            file_extension = submitted_file.filename.rsplit('.', 1)[1].lower()
            
            # build path
            path = os.path.join(uploadsfolder,time.strftime("%Y%m%d-%H%M%S") + '.' + file_extension)
            
            # save file
            submitted_file.save(path)
            

            ##############
            ### do prediction
            ##############
            batches, preds = vgg.test(test_path, batch_size = batch_size*2)

            #############
            ### build output
            #############
            
            # get filenames and index of this image
            filenames = batches.filenames
            filenames = [f.replace("unknown/","") for f in filenames]
            #
            splits = path.split("/")
            fname = splits[len(splits) -1]
            print 'filename', fname
            print (" ; ").join(filenames)
            #
            f_idx = filenames.index(fname)

            # round probabilities, convert to list, pull prediction for just this image
            probs = np.round(preds,decimals = 4)*100
            probslist = probs.tolist()
            probs = probslist[f_idx]

            ### create output object
            output = {}
            output['cat'] = probs[0]
            output['dog'] = probs[1]


            ######################
            ### move processed image
            ######################

            # move image now that it's been processed
            path_new = path.replace("_uploads/unknown","_uploads_done")
            print 'moving processed image from:'
            print path
            print 'to:'
            print path_new
            os.rename(path,path_new)

            # output path to uploaded image (need to strip absolute portions of path)
            image_path = "static/" + path_new.split("/static/")[1]

            output = json.dumps(output)

            return render_template('result.html', dog = probs[1], cat=probs[0], image_path = image_path)
    else:       
        return render_template('error.html')


#######
### RUN
#######
if __name__ == "__main__":

    # run app
    app.run(host = "0.0.0.0", port = int("5000"))