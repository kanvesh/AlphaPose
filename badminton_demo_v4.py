from pose_utils import *
import pickle
import sys
import os
import glob
import numpy as np
import pandas as pd
from annotator import Annotator
import more_itertools as mit
import time
import pickle

sys.path.append('../code/')
from utils import *

url = sys.argv[1]
start_time = sys.argv[2]
duration = sys.argv[3]
sample_fps=2

#url = 'https://www.youtube.com/watch?v=IAoIhhbCxvE'
#start_time = '00:30:00'
#duration = '00:02:00'

os.system('rm -rf temp')
os.system('mkdir temp')

print('Downloading video')
os.system('youtube-dl -f 133 '+url+' --output temp/temp.mp4')

print('Cutting video between timestamps')
os.system('ffmpeg -i temp/temp.mp4 -ss '+start_time+' -t '+duration+' -c copy temp/temp_clipped.mp4')

os.system('mkdir temp/temp_images')
os.system('ffmpeg -i temp/temp_clipped.mp4 -r '+str(sample_fps)+' temp/temp_images/output_%04d.png')
#os.system('python demo.py --indir temp/temp_images/ --outdir temp --sp')

#os.mkdir('temp/temp_clips')
#ann= Annotator(None,None)
#ann.video_to_clips(video_file='temp/temp_clipped.mp4',output_folder='temp/temp_clips',clip_length=60)

os.system('python demo.py --indir temp/temp_images/ --outdir temp  --conf 0.5 --nms 0.45 --sp --detbatch 2')

df = pd.read_json('temp/alphapose-results.json')
df['clip']=[np.floor(int(i.split('.')[0].split('_')[-1])) for i in df.image_id]
df['normalized_tuples'] = df.keypoints.apply(normalize_tuples)
df['normalized_keypoints'] = df.keypoints.apply(normalize_keypoints)
df['width'] = df.keypoints.apply(get_width)
df['height'] = df.keypoints.apply(get_height)
df['angle_at_leftelbow'] = df.keypoints.apply(get_angle_at_joint, args=('LShoulder','LElbow','LWrist'))
df['angle_at_rightelbow'] = df.keypoints.apply(get_angle_at_joint, args=('RShoulder','RElbow','RWrist'))
df['angle_at_leftshoulder'] = df.keypoints.apply(get_angle_at_joint, args=('LHip','LShoulder','LElbow'))
df['angle_at_rightshoulder'] = df.keypoints.apply(get_angle_at_joint, args=('RHip','RShoulder','RElbow'))
df['angle_at_leftknee'] = df.keypoints.apply(get_angle_at_joint, args=('LHip','LKnee','LAnkle'))
df['angle_at_rightknee'] = df.keypoints.apply(get_angle_at_joint, args=('RHip','RKnee','RAnkle'))
df['angle_at_hip'] = df.keypoints.apply(get_angle_at_hip)
df['is_person_standing'] = df.keypoints.apply(is_person_standing_straight)

df=  df.sort_values(by=['image_id', 'height'],ascending=False).drop_duplicates(['image_id'])

X_mat1 = np.vstack( (df.normalized_keypoints))
X_mat2 = np.matrix(df[['angle_at_leftelbow','angle_at_rightelbow','angle_at_leftshoulder','angle_at_rightshoulder','angle_at_leftknee','angle_at_rightknee','angle_at_hip']])
X_mat = np.hstack((X_mat1,X_mat2))


print('Loading Model')
clf = pickle.load(open('custom_code/pose_model_v6.p','rb'))

pred = clf.predict(X_mat)
pred_proba = clf.predict_proba(X_mat)

df['pred']= pred
df['pred_proba']= pred_proba[:,1]
res = df.groupby(['clip']).pred_proba.mean()
moving_res = [np.sum(res[i-2:i+3])/5 for i in range(len(res))]
print(res)
print(moving_res)

threshold = np.percentile(moving_res,50,interpolation='nearest')
indices = np.where(moving_res>=threshold)[0]

#if clips are missing with small differences between start times, need to fill them up
indices=list(indices)
indices_new=[]
for i in range(len(indices)-1):
    indices_new.append(indices[i])
    if (indices[i+1]-indices[i])==2:
        indices_new.append(indices[i]+1)
                           
indices_new.append(indices[-1])


groups = [list(group) for group in mit.consecutive_groups(indices_new)]

log = {"url":url, "start_time":start_time, "duration":duration, "df":df, "sample_fps":sample_fps, "model":clf, "result":res, "moving_res":moving_res, "threshold":threshold, "indices":indices, "groups":groups}
pickle.dump(log,open('temp/log.p','wb'))


os.system('mkdir temp/temp_clips')

f= open("clipslist.txt","w+")

for group in groups:
    
    ##prev noplay clip
    
    try:
        start_time = frame_to_timestamp(prev_group[-1], sample_fps)
        duration = frame_to_timestamp(group[0]-prev_group[-1], sample_fps)
        print(start_time, duration)
        os.system('ffmpeg -y -ss '+start_time+' -i temp/temp_clipped.mp4 -strict -2 -t '+duration+' temp/temp_clips/out'+str(prev_group[-1])+'.mp4')
        os.system('ffmpeg -y -i temp/temp_clips/out'+str(prev_group[-1])+'.mp4 -vf drawtext="text=noplay: fontcolor=black: fontsize=24: box=1: boxcolor=white@0.8: boxborderw=5: x=(w-text_w)/2: y=(text_h)/1.1" -codec:a copy temp/temp_clips/out'+str(prev_group[-1])+'_with_label.mp4')
        os.system('ffmpeg -i temp/temp_clips/out'+str(prev_group[-1])+'_with_label.mp4 -strict -2 -filter:v "setpts=0.25*PTS" temp/temp_clips/out'+str(prev_group[-1])+'_with_label_spedup.mp4')
        f.write("file "+"temp/temp_clips/out"+str(prev_group[-1])+"_with_label_spedup.mp4\n")
    except:
        pass
    
    ##Current play clip
    if len(group)>2:
        dptext='play'
    else:
        dptext='noplay'
        
    start_time = frame_to_timestamp(group[0], sample_fps)
    duration = frame_to_timestamp(group[-1]-group[0], sample_fps)
    os.system('ffmpeg -y -ss '+start_time+' -i temp/temp_clipped.mp4 -strict -2 -t '+duration+' temp/temp_clips/out'+str(group[0])+'.mp4')
    os.system('ffmpeg -y -i temp/temp_clips/out'+str(group[0])+'.mp4 -vf drawtext="text='+dptext+': fontcolor=black: fontsize=24: box=1:boxcolor=white@0.8: boxborderw=5: x=(w-text_w)/2: y=(text_h)/1.1" -codec:a copy temp/temp_clips/out'+str(group[0])+'_with_label.mp4')
    f.write("file "+"temp/temp_clips/out"+str(group[0])+"_with_label.mp4\n")
    #f.write("file "+"temp/temp_clips/out"+str(group[0])+".mp4\n")
    
    
    prev_group = group
    
        
    
        
f.close()


os.system('ffmpeg -f concat -safe 0 -i clipslist.txt -c copy temp/output.mp4')
os.system('ffmpeg -i temp/output.mp4 -strict -2 -filter:v "setpts=0.5*PTS" temp/output_spedup.mp4')


#ffmpeg -y -i output_spedup.mp4 -vf drawtext="text='play': fontcolor=black: fontsize=24: box=1: boxcolor=white@0.8: boxborderw=5: x=(w-text_w)/2: y=(text_h)/1.1" -codec:a copy output_with_label.mp4

rand=int(time.time())
os.system('mkdir results/run_'+str(rand))


#os.system('rm clipslist.txt')

os.system('mv temp/* results/run_'+str(rand))
os.system('rm -r temp')
print(threshold)