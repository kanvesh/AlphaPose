######################################################################################################

import math
import os
import scipy.io.wavfile as saudio
import numpy as np
from scipy import fft, arange
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import cv2
from PIL import Image

######################################################################################################

def clip_to_images(cliplocation, path_to_store_images,images_per_second=5):       
    """
    Saves screenshots from clips at a specified rate
    """       
    try:
        os.system('mkdir '+path_to_store_images)
    except:
        pass
    
    result = os.system('ffmpeg -i '+cliplocation+' -vf fps='+str(images_per_second)+" "+path_to_store_images+'out%d.png')
    if result==0:
        print('Success')
    else:
        print('Something went wrong')
    return result

######################################################################################################

def frame_to_timestamp(frame,fps):
    """
    Converts frame number to a timestamp based on the video's FPS
    """
    tot_secs = frame/fps
    return secs_to_timestamp(tot_secs)

######################################################################################################

def secs_to_timestamp(tot_secs):
    """
    Represents seconds in a hh:mm:ss timestamp format
    """
    hours = math.floor(tot_secs/3600)
    mins = math.floor((tot_secs/60)%60)
    secs = tot_secs%60
    return str(hours).zfill(2)+':'+str(mins).zfill(2)+':'+str(round(secs,4)).zfill(2)

######################################################################################################

def get_predicted_classes(pred_vectors, classes, apply_softmax=False):       
    """
    Takes in an array of predicted vectors (for a batch of inputs) and returns 
    the corresponding class outputs along with proportions (equivalent to probabilities) 
    """    
    if apply_softmax:
        pred_vectors = [softmax(vector) for vector in pred_vectors]
        
    pred_classes = [classes[vec.argmax()] for vec in pred_vectors]
    class_counts = pd.Series(pred_classes).value_counts()
    class_proporions = [(class_counts.index[0],1.0*class_counts[0]/sum(class_counts)) for i in class_counts]
    
    return class_proportions

######################################################################################################

def majority_class(pred_vectors, classes, cutoff, apply_softmax=False):
    """
    Takes in an array of predicted vectors (for a batch of inputs) and returns predicted classes 
    if the 'probability' crosses cutoff, else returns 'Confused'
    """
    class_proportions = get_predicted_classes(pred_vectors, classes, cutoff, apply_softmax)
    
    if class_proportions[0][1]>cutoff:
        return class_proportions[0][0]
    else:
        return 'Confused'
    
######################################################################################################

def predict_image_batch(tensor_concat, model, classes):      
    """
    Runs a bunch of tensor inputs through a model and returns output vectors
    """
    outp = model(tensor_concat)
    vectors = [i.data.cpu().numpy() for i in outp]
    return vectors

######################################################################################################

def image_transform(frame,scale=(224,224)):
    """
    converts image to tensor input
    """
    frame = cv2.resize(frame,(224,224))
    pilimage = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = test_transforms(pilimage).float()
    image_tensor = image_tensor.unsqueeze_(0)
    inp = Variable(image_tensor)
    inp = inp.to(device)
    return inp
              
######################################################################################################

def audio_to_array(mp4file, freq=44100, channels=1):
    """
    Takes in an mp4 video file and generates an audio array in time domain.
    Returns as many arrays as number of channels specified. You can also set frequency. 
    """
    os.system('ffmpeg -i '+mp4file+' -ab 160k -ac '+str(channels)+' -ar '+str(freq)+' -vn -y temp.wav')
    arr= saudio.read('temp.wav')
    os.system('rm temp.wav')
    return arr[1]

######################################################################################################

def frequency_spectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    print(n)
    k = arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)

######################################################################################################

def softmax(arr):
    arr = [math.exp(i) for i in arr]
    return arr/np.sum(arr)

######################################################################################################
