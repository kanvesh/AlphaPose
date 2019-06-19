#!/usr/bin/env python
# coding: utf-8

# In[4]:


from pose_utils import *
import pickle


# In[5]:


clf = pickle.load(open('pose_model_v1.p','rb'))


# In[ ]:


df_concat=pd.DataFrame()
for i in os.listdir('../output/badm001/'):
    try:
        print(i)
        path = '../output/badm001/'+i+'/alphapose-results.json'
        df = pd.read_json(path)
        df['clip']=str(i)
        df['normalized_tuples'] = df.keypoints.apply(normalize_tuples)
        df['normalized_keypoints'] = df.keypoints.apply(normalize_keypoints)
        df['width'] = df.keypoints.apply(get_width)
        df['height'] = df.keypoints.apply(get_height)
        df['angle_at_hip'] = df.keypoints.apply(get_angle_at_hip)
        df['is_person_standing'] = df.keypoints.apply(is_person_standing_straight)
        df_concat = df_concat.append(df)
        print(df_concat.shape)
    except:
        pass


# In[ ]:


df_concat= df_concat.sort_values(by=['image_id', 'height'],ascending=False).drop_duplicates(['clip','image_id'])


# In[ ]:


X_mat = np.vstack( (df_concat.normalized_keypoints))


# In[ ]:


pred = clf.predict(X_mat)


# In[ ]:


df_concat['pred']= pred
res = df_concat.groupby(['clip']).pred.mean()


# In[ ]:


f= open("clipslist.txt","w+")

for i in list(res.index[res>0.1]):
    print(i)
    try:
        f.write("file "+"../output/clips_badminton001/"+i+".mp4\n")
    except:
        pass
f.close()

os.system('ffmpeg -f concat -safe 0 -i clipslist.txt -c copy highlights.mp4')

os.system('rm clipslist.txt')


# In[ ]:




