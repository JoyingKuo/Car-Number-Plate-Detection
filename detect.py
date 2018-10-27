#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines to detect number plates.

Use `detect` to detect all bounding boxes, and use `post_process` on the output
of `detect` to filter using non-maximum suppression.

"""


__all__ = (
    'detect',
    'post_process',
    'letter_probs_to_code',
    'make_scaled_ims',
    '_overlaps',
    '_group_overlapping_rectangles',
    'post_process',    
#    'code',
)


import collections
import itertools
import math
import sys

import cv2
import numpy
import tensorflow as tf

import common
import model

import string
import random
import pyrebase
import subprocess
import os



def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))


#def detect(im, param_vals):
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
    #scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))

    # Load the model which detects number plates over a sliding window.
    #x, y, params = model.get_detect_model()

    # Execute the model at each scale.
    #with tf.Session(config=tf.ConfigProto()) as sess:
      #  y_vals = []
       # for scaled_im in scaled_ims:
        #    feed_dict = {x: numpy.stack([scaled_im])}
         #   feed_dict.update(dict(zip(params, param_vals)))
          #  y_vals.append(sess.run(y, feed_dict=feed_dict))

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    #for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
      #  for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                      # -math.log(1./0.99 - 1)):
           # letter_probs = (y_val[0,
                                 # window_coords[0],
                                #  window_coords[1], 1:].reshape(
                                 #   7, len(common.CHARS)))#letter_prob should be random
            #
            #print "letter_probs1 in detect()"
            #print letter_probs 
            #  
            #letter_probs = common.softmax(letter_probs)
            #
            #print "letter_probs2 in detect()"
            #print letter_probs  
            #
 
            #img_scale = float(im.shape[0]) / scaled_im.shape[0]

            #bbox_tl = window_coords * (8, 4) * img_scale
           # bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

           # present_prob = common.sigmoid(
               #                y_val[0, window_coords[0], window_coords[1], 0])

          #  yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs


def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
        else:
            match_to_group[idx1] = num_groups 
            num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups


def post_process(matches):
    """
    Take an iterable of matches as returned by `detect` and merge duplicates.

    Merging consists of two steps:
      - Finding sets of overlapping rectangles.
      - Finding the intersection of those sets, along with the code
        corresponding with the rectangle with the highest presence parameter.

    """
    groups = _group_overlapping_rectangles(matches)
    #print "groups"
    #print groups
    for group_matches in groups.values():
        mins = numpy.stack(numpy.array(m[0]) for m in group_matches)
        #print "mins"
        #print mins
        maxs = numpy.stack(numpy.array(m[1]) for m in group_matches)
        #print "maxs"
        #print maxs
        present_probs = numpy.array([m[2] for m in group_matches])
        #print "present_probs"
        #print present_probs
        letter_probs = numpy.stack(m[3] for m in group_matches)
        #print "letter_probs"
        #print letter_probs
        yield (numpy.max(mins, axis=0).flatten(),
               numpy.min(maxs, axis=0).flatten(),
               numpy.max(present_probs),
               letter_probs[numpy.argmax(present_probs)])


def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))

def id_generator(size=18, chars = string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

#if __name__ == "__main__":
    #contact with firebase
    #config = {
    #  "apiKey": "AIzaSyAyu0tIQgWiU13sF6NBsdhFHy3FdU9oHLY",
    #  "authDomain": "car-number-plate.firebaseapp.com",
    #  "databaseURL": "https://car-number-plate.firebaseio.com",
    #  "storageBucket": "car-number-plate.appspot.com",
    #  "serviceAccount": "/lpr/deep-anpr/car-number-plate-firebase-adminsdk-x0gm2-7601fd9d3d.json"
    #}
    
    #firebase = pyrebase.initialize_app(config)
    ###authentication
    #auth = firebase.auth()
    #user = auth.sign_in_with_email_and_password("nctucscar108@gmail.com", "CAR108car108")
    #db = firebase.database() 
    ##db.child("return_number")	
    #storage = firebase.storage()
	
    #im = cv2.imread(sys.argv[1])
    #im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.
    #im_gray2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) /2
    #cv2.imwrite("input.jpg", im_gray2)
    #im = cv2.imread("input.jpg")
    #f = numpy.load(sys.argv[2])
    #param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    #for pt1, pt2, present_prob, letter_probs in post_process(
                                                  #detect(im_gray, param_vals)):
      #  pt1 = tuple(reversed(map(int, pt1)))
       # pt2 = tuple(reversed(map(int, pt2)))
       # tmp1 = (pt1[0]+pt2[0])/2
       # tmp2 = (pt1[1]+pt2[1])/2
        #
       # print "### pt1 and pt2"
       # print pt1
       # print pt2
        #

        #
        #print "### letter_probs in main:"
        #print letter_probs
        #
       # code = letter_probs_to_code(letter_probs)
        #
        #print "### code= "
        #print code
        #
        #out_prob =[]
        #out_prob = (numpy.max(letter_probs, axis=1))
        #print len(out_prob)
        #print "### out_prob= "
        #print sum(out_prob)
        #
        #code_index =[]
        #code_index = (numpy.argmax(letter_probs, axis=1))
        #print "### code_index= "
        #print code_index

        #color = (B, G, R)
        #color = (0.0, 255.0, 0.0)
       # color = (0.0, 0.0, 255.0)
        #cv2.rectangle(im, pt1, pt2, color)

        #cv2.putText(im,
        #            code,
        #            pt1,
        #            cv2.FONT_HERSHEY_PLAIN, 
        #            1.5,
        #            (0, 0, 255),
        #            thickness=5)

        #cv2.putText(im,
        #            code,
        #            pt1,
         #           cv2.FONT_HERSHEY_PLAIN, 
         #           1.5,
         #           (255, 255, 255),
         #           thickness=2)

        #write left-top position of license plate with license plate numbers
        #outcode = "P(%s,%s): " % (tmp1,tmp2) + code

        #cv2.putText(im,
        #            outcode,
        #            pt1,
        #            cv2.FONT_HERSHEY_PLAIN, 
        #            1.5,
        #            (0, 0, 255),
        #            thickness=5)

        #cv2.putText(im,
        #            outcode,
        #            pt1,
        #            cv2.FONT_HERSHEY_PLAIN, 
        #            1.5,
        #            (255, 255, 255),
        #            thickness=2)

    #cv2.imwrite(sys.argv[3], im)
    
    #os.rename(sys.argv[3],code+".jpg")
    #return_image= "result/" + code + ".jpg"
    #storage.child(return_image).put(code + ".jpg")
    #lenG=len(sys.argv[1])
    #print lenG
    #print sys.argv[1][0:lenG-4]
    
    #reuturn_str=sys.argv[1][0:lenG-4]+"_return"
    #cv2.imwrite("gray-" + sys.argv[3], im_gray)
    #data = {"%s"%(reuturn_str): "%s"%(code)}
    #db.child("%s"%(reuturn_str)).set(data)
    #db.push(data)
	
    

