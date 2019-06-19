#!/usr/bin/env python
# coding: utf-8

# In[25]:


from pose_utils import *
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import confusion_matrix
import pickle


# In[23]:


df_concat=pd.DataFrame()
for i in os.listdir('/mnt/disks/cricket-highlights/anvesh/AlphaPose/output/badm004/'):
    print(i)
    path = '/mnt/disks/cricket-highlights/anvesh/AlphaPose/output/badm004/'+i+'/alphapose-results.json'
    df = pd.read_json(path)
    df['clip']=str(i)
    df['normalized_tuples'] = df.keypoints.apply(normalize_tuples)
    df['normalized_keypoints'] = df.keypoints.apply(normalize_keypoints)
    df['width'] = df.keypoints.apply(get_width)
    df['height'] = df.keypoints.apply(get_height)
    df['angle_at_hip'] = df.keypoints.apply(get_angle_at_hip)
    df['is_person_standing'] = df.keypoints.apply(is_person_standing_straight)
    df_concat = df_concat.append(df)


# In[12]:


res = pd.read_json('../output/badminton004.json')
play_clips = [i.split('\\')[-1].split('.')[0] for i in res.video]


# In[13]:


df_concat['play'] = [1 if i in play_clips else 0 for i in df_concat['clip']  ]


# In[14]:


# Stores only the person with max height. All other persons in the image are dropped
df_concat= df_concat.sort_values(by=['image_id', 'height'],ascending=False).drop_duplicates(['clip','image_id'])


# In[16]:


X_mat = np.vstack( (df_concat.normalized_keypoints))
y= df_concat.play


# In[17]:


gss = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=13)
data_split = gss.split(X_mat, y, groups = df_concat['clip'])
train_ids, test_ids = next(data_split)
X_train, X_test, y_train, y_test = X_mat[train_ids],X_mat[test_ids],np.array(y)[train_ids],np.array(y)[test_ids] 


# In[18]:


clf = rfc()
clf.fit(X_train,y_train)


# In[19]:


pred = clf.predict(X_test)


# In[22]:


print(confusion_matrix(y_test,pred))


# In[26]:


pickle.dump(clf, open('pose_model_v1.p','wb'))


# In[ ]:




