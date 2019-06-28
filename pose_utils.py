import pandas as pd
import numpy as np
import os
import math

coco_format ={   0:"Nose",
    1:"LEye",
    2:"REye",
    3:"LEar",
    4:"REar",
    5:"LShoulder",
    6:  "RShoulder",
    7:  "LElbow",
    8:  "RElbow",
    9:  "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "Rknee",
    15: "LAnkle",
    16: "RAnkle"}

coco_format_flip = {v:k for k,v in coco_format.items()}

def get_3_tuples(keypoints):
    tuples=[]
    for i in range(int(len(keypoints)/3)):
        tuples.append((keypoints[3*i],keypoints[3*i+1],keypoints[3*i+2]))
    return tuples


def get_width(keypoints):
    tuples = get_3_tuples(keypoints)
    xs = [i[0] for i in tuples]
    return np.max(xs) - np.min(xs)

def get_height(keypoints):
    tuples = get_3_tuples(keypoints)
    ys = [i[1] for i in tuples]
    return np.max(ys) - np.min(ys)


def normalize_tuples(keypoints):
    tuples = get_3_tuples(keypoints)
    xs = [i[0] for i in tuples]
    ys = [i[1] for i in tuples]
    centre = ((np.max(xs)+np.min(xs))/2,(np.max(ys)+np.min(ys))/2)
    for i in range(len(tuples)):
        tuples[i]=(tuples[i][0]-centre[0],tuples[i][1]-centre[1],tuples[i][2])

    return tuples

def normalize_keypoints(keypoints):
    tuples = get_3_tuples(keypoints)
    xs = [i[0] for i in tuples]
    ys = [i[1] for i in tuples]
    centre = ((np.max(xs)+np.min(xs))/2,(np.max(ys)+np.min(ys))/2)
    for i in range(len(tuples)):
        tuples[i]=(tuples[i][0]-centre[0],tuples[i][1]-centre[1],tuples[i][2])

    return [i for tuple in tuples for i in tuple]

def get_part_position(bodypart, keypoints, normalize=True):
    tuples = get_3_tuples(keypoints)
    if normalize:
        tuples = normalize_tuples(keypoints)
    try:
        return tuples[coco_format_flip[bodypart]][0:2]
    except:
        print('Something wrong')
        
def get_angle_vector(v1,v2):
    x1=v1[0]
    y1=v1[1]
    x2=v2[0]
    y2=v2[1]
    angle = 180*(math.atan2(y1,x1)-math.atan2(y2,x2))/3.1417
    return abs(angle)

def get_angle_at_hip(keypoints):
    nose = get_part_position('Nose',keypoints)
    LHip = get_part_position('LHip',keypoints)
    LAnkle = get_part_position('LAnkle',keypoints)
    RHip = get_part_position('RHip',keypoints)
    RAnkle = get_part_position('RAnkle',keypoints)
                               
    v1= tuple(np.subtract(nose,LHip))
    v2= tuple(np.subtract(LHip,LAnkle))
    left_angle = get_angle_vector(v1,v2)
                               
    v1= tuple(np.subtract(nose,RHip))
    v2= tuple(np.subtract(RHip,RAnkle))
    right_angle = get_angle_vector(v1,v2)
    return left_angle+right_angle

def is_person_standing_straight(keypoints):
    if get_angle_at_hip(keypoints)<60:
        return 1
    else:
        return 0