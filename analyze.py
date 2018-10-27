import collections
import itertools
import math
import sys

import cv2
import numpy
import tensorflow as tf

import common
import model
import detect


import string
import random
import pyrebase
import subprocess
import os
#import detect


#config = {
#  "apiKey": "AIzaSyAyu0tIQgWiU13sF6NBsdhFHy3FdU9oHLY",
#  "authDomain": "car-number-plate.firebaseapp.com",
#  "databaseURL": "https://car-number-plate.firebaseio.com",
#  "storageBucket": "car-number-plate.appspot.com",
#  "serviceAccount": "/home/joying/car-number-plate-firebase-adminsdk-x0gm2-7601fd9d3d.json"
#}

#firebase = pyrebase.initialize_app(config)
in_6=0
in_7=0
in_8=0
##authentication
#auth = firebase.auth()
#user = auth.sign_in_with_email_and_password("nctucscar108@gmail.com", "CAR108car108")
#db = firebase.database()

#data = {"key": 1}
#db.push(data)
#db.child("car-number-plate").child("image_name")
#data = {"name": "Mortimer 'Morty' Smith"}
#data = {"num": 1}
#db.child("car-number-plate").push(data)
#users=db.child("image_name").shallow().get()
#print users

#db.child("users").child("image").set(data)
#results = db.child("users").push(data, user['idToken'])

#storage = firebase.storage()

#####get detect_model6####################
def get_detect_model_6():
  x, conv_layer, conv_vars = model.convolutional_layers()
    
# Fourth layer
  W_fc1 = model.weight_variable([8 * 32 * 64, 1024])
  W_conv1 = tf.reshape(W_fc1, [8,  32, 64, 1024])
  b_fc1 = model.bias_variable([1024])
  h_conv1 = tf.nn.relu(model.conv2d(conv_layer, W_conv1,
                            stride=(1, 1), padding="VALID") + b_fc1)
# Fifth layer
  W_fc2 = model.weight_variable([1024, 1 + 6 * len(common.CHARS)])
  W_conv2 = tf.reshape(W_fc2, [1, 1, 1024, 1 + 6 * len(common.CHARS)])
  b_fc2 = model.bias_variable([1 + 6 * len(common.CHARS)])
  h_conv2 = model.conv2d(h_conv1, W_conv2) + b_fc2

  return (x, h_conv2, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])
  
  
#####get detect_model7####################
def get_detect_model_7():
  x, conv_layer, conv_vars = model.convolutional_layers()
    
# Fourth layer
  W_fc1 = model.weight_variable([8 * 32 * 64, 1024])
  W_conv1 = tf.reshape(W_fc1, [8,  32, 64, 1024])
  b_fc1 = model.bias_variable([1024])
  h_conv1 = tf.nn.relu(model.conv2d(conv_layer, W_conv1,
                            stride=(1, 1), padding="VALID") + b_fc1)
# Fifth layer
  W_fc2 = model.weight_variable([1024, 1 + 7 * len(common.CHARS)])
  W_conv2 = tf.reshape(W_fc2, [1, 1, 1024, 1 + 7 * len(common.CHARS)])
  b_fc2 = model.bias_variable([1 + 7 * len(common.CHARS)])
  h_conv2 = model.conv2d(h_conv1, W_conv2) + b_fc2

  return (x, h_conv2, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])
  
  
#####get detect_mode8####################
def get_detect_model_8():
  x, conv_layer, conv_vars = model.convolutional_layers()
    
# Fourth layer
  W_fc1 = model.weight_variable([8 * 32 * 64, 1024])
  W_conv1 = tf.reshape(W_fc1, [8,  32, 64, 1024])
  b_fc1 = model.bias_variable([1024])
  h_conv1 = tf.nn.relu(model.conv2d(conv_layer, W_conv1,
                            stride=(1, 1), padding="VALID") + b_fc1)
# Fifth layer
  W_fc2 = model.weight_variable([1024, 1 + 8 * len(common.CHARS)])
  W_conv2 = tf.reshape(W_fc2, [1, 1, 1024, 1 + 8 * len(common.CHARS)])
  b_fc2 = model.bias_variable([1 + 8 * len(common.CHARS)])
  h_conv2 = model.conv2d(h_conv1, W_conv2) + b_fc2

  return (x, h_conv2, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])

###################detect_fun6###############

def detect_6(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.

    """

    # Convert the image to various scales.
    scaled_ims = list(detect.make_scaled_ims(im, model.WINDOW_SHAPE))

    # Load the model which detects number plates over a sliding window.
    x, y, params = get_detect_model_6()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)):
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    6, len(common.CHARS)))#letter_prob should be random
            #
            #print "letter_probs1 in detect()"
            #print letter_probs 
            #  
            letter_probs = common.softmax(letter_probs)
            #
            #print "letter_probs2 in detect()"
            #print letter_probs  
            #
 
            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

            present_prob = common.sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0])

            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs


###################detect_fun7###############
def detect_7(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.

    """

    # Convert the image to various scales.
    scaled_ims = list(detect.make_scaled_ims(im, model.WINDOW_SHAPE))

    # Load the model which detects number plates over a sliding window.
    x, y, params = get_detect_model_7()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)):
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    7, len(common.CHARS)))#letter_prob should be random
            #
            #print "letter_probs1 in detect()"
            #print letter_probs 
            #  
            letter_probs = common.softmax(letter_probs)
            #
            #print "letter_probs2 in detect()"
            #print letter_probs  
            #
 
            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

            present_prob = common.sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0])

            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs
            
            
###################detect_fun8###############
def detect_8(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.

    """

    # Convert the image to various scales.
    scaled_ims = list(detect.make_scaled_ims(im, model.WINDOW_SHAPE))

    # Load the model which detects number plates over a sliding window.
    x, y, params = get_detect_model_8()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)):
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    8, len(common.CHARS)))#letter_prob should be random
            #
            #print "letter_probs1 in detect()"
            #print letter_probs 
            #  
            letter_probs = common.softmax(letter_probs)
            #
            #print "letter_probs2 in detect()"
            #print letter_probs  
            #
 
            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

            present_prob = common.sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0])

            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs


#############detect main#####################

if __name__ == "__main__":	
    im = cv2.imread(sys.argv[1])
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.
    
    
    
#####################for 6####################
    global in_6,in_7,in_8
    f_6 = numpy.load(sys.argv[2])
    param_vals_6 = [f_6[n] for n in sorted(f_6.files, key=lambda s: int(s[4:]))]
    

    for pt6_1, pt6_2, present_prob_6, letter_probs_6 in detect.post_process(
                                                  detect_6(im_gray, param_vals_6)):
        in_6=1
        pt6_1 = tuple(reversed(map(int, pt6_1)))
        pt6_2 = tuple(reversed(map(int, pt6_2)))

        #
        print "### pt6_1 and pt6_2"
        print pt6_1
        print pt6_2
        #

        #
        #print "### letter_probs in main:"
        #print letter_probs
        #
        code_6 = detect.letter_probs_to_code(letter_probs_6)
        #
        #print "### code= "
        #print code
        #
        out_prob_6 =[]
        out_prob_6 = (numpy.max(letter_probs_6, axis=1))
        #print len(out_prob_6)
        print "### out_prob_6= "
        print sum(out_prob_6)
        cmp_6= sum(out_prob_6)/6
        print cmp_6
        #
        code_index_6 =[]
        code_index_6 = (numpy.argmax(letter_probs_6, axis=1))
        #print "### code_index= "
        #print code_index

        #color = (B, G, R)
        #color = (0.0, 255.0, 0.0)
    if in_6==0 :
       cmp_6=0    



#####################for 7####################
    f_7 = numpy.load(sys.argv[3])
    param_vals_7 = [f_7[n] for n in sorted(f_7.files, key=lambda s: int(s[4:]))]
    for pt7_1, pt7_2, present_prob_7, letter_probs_7 in detect.post_process(
                                                  detect_7(im_gray, param_vals_7)):
        in_7=1
        print 123
        pt7_1 = tuple(reversed(map(int, pt7_1)))
        pt7_2 = tuple(reversed(map(int, pt7_2)))

        #
        print "### pt7_1 and pt7_2"
        print pt7_1
        print pt7_2
        #

        #
        #print "### letter_probs in main:"
        #print letter_probs
        #
        code_7 = detect.letter_probs_to_code(letter_probs_7)
        #
        #print "### code= "
        #print code
        #
        out_prob_7 =[]
        out_prob_7 = (numpy.max(letter_probs_7, axis=1))
        print len(out_prob_7)
        print "### out_prob_7= "
        print sum(out_prob_7)
        cmp_7= sum(out_prob_7)/7
        print cmp_7
        #
        code_index_7 =[]
        code_index_7 = (numpy.argmax(letter_probs_7, axis=1))
        #print "### code_index= "
        #print code_index

        #color = (B, G, R)
        #color = (0.0, 255.0, 0.0)
    if in_7==0 :
       cmp_0=0 
        
######################## for 8################
    f_8 = numpy.load(sys.argv[4])
    param_vals_8 = [f_8[n] for n in sorted(f_8.files, key=lambda s: int(s[4:]))]
    for pt8_1, pt8_2, present_prob_8, letter_probs_8 in detect.post_process(
                                                  detect_8(im_gray, param_vals_8)):
        in_8=1
        print 123
        pt8_1 = tuple(reversed(map(int, pt8_1)))
        pt8_2 = tuple(reversed(map(int, pt8_2)))

        #
        print "### pt8_1 and pt8_2"
        print pt8_1
        print pt8_2
        #

        #
        #print "### letter_probs in main:"
        #print letter_probs
        #
        code_8 = detect.letter_probs_to_code(letter_probs_8)
        #
        #print "### code= "
        #print code
        #
        out_prob_8 =[]
        out_prob_8 = (numpy.max(letter_probs_8, axis=1))
        print len(out_prob_8)
        print "### out_prob_8= "
        print sum(out_prob_8)
        cmp_8= sum(out_prob_8)/8 ############## need to retraining
        print cmp_8
        #
        code_index_8 =[]
        code_index_8 = (numpy.argmax(letter_probs_8, axis=1))
        #print "### code_index= "
        #print code_index

        #color = (B, G, R)
        #color = (0.0, 255.0, 0.0)
    if in_8==0 :
     cmp_8=0
     
    if cmp_6==max(cmp_6,cmp_7,cmp_8):
       color = (0.0, 0.0, 255.0)
       cv2.rectangle(im, pt6_1, pt6_2, color)

        #write left-top position of license plate with license plate numbers
       tmp1 = (pt6_1[0]+pt6_2[0])/2
       tmp2 = (pt6_1[1]+pt6_2[1])/2
       outcode = "P(%s,%s): " % (tmp1,tmp2) + code_6

       cv2.putText(im,
                    outcode,
                    pt6_1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (0, 0, 255),
                    thickness=5)

       cv2.putText(im,
                    outcode,
                    pt6_1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (255, 255, 255),
                    thickness=2)
    
       cv2.imwrite(sys.argv[5], im)
       
    elif cmp_7==max(cmp_6,cmp_7,cmp_8):
       color = (0.0, 0.0, 255.0)
       cv2.rectangle(im, pt7_1, pt7_2, color)

        #write left-top position of license plate with license plate numbers
       tmp1 = (pt7_1[0]+pt7_2[0])/2
       tmp2 = (pt7_1[1]+pt7_2[1])/2
       outcode = "P(%s,%s): " % (tmp1,tmp2) + code_7

       cv2.putText(im,
                    outcode,
                    pt7_1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (0, 0, 255),
                    thickness=5)

       cv2.putText(im,
                    outcode,
                    pt7_1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (255, 255, 255),
                    thickness=2)
    
       cv2.imwrite(sys.argv[5], im)
    
    else :
       color = (0.0, 0.0, 255.0)
       cv2.rectangle(im, pt8_1, pt8_2, color)

        #write left-top position of license plate with license plate numbers
       tmp1 = (pt8_1[0]+pt8_2[0])/2
       tmp2 = (pt8_1[1]+pt8_2[1])/2
       outcode = "P(%s,%s): " % (tmp1,tmp2) + code_8

       cv2.putText(im,
                    outcode,
                    pt8_1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (0, 0, 255),
                    thickness=5)

       cv2.putText(im,
                    outcode,
                    pt8_1,
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5,
                    (255, 255, 255),
                    thickness=2)
    
       cv2.imwrite(sys.argv[5], im)
    
        
#######################################################
def id_generator(size=18, chars = string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


  
