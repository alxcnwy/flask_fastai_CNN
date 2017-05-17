### Flask app wrapper for convolutional neural network (CNN) model covered in lesson 2 of the fast.ai deep learning MOOC 

Allows the user to upload an image then that image is fed through a precomputed CNN (the weights are loaded from static/_model/modelweights.h5 ) and the predictions are returned on the next page. Works for 2 classes but easy to extend to arbitrary number of classes given a pre-trained model. 

To run just do:
python server.py 

Enjoy!