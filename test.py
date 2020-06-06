import threading
#import threading.thread
import queue
import multiprocessing as mp
import sys
global stop_threads
import tensorflow as tf
import pandas as pd
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
model_final=load_model("model_alexnet_best2.h5")
import predict1
import attributes


stop_threads=False
num=input('do u want to start')
if num=='y' :
    from PIL import ImageGrab
    import cv2
    import time
    import numpy as np
    def trial(q,threadName):
        #x=[]
        #i=1
        while True:
            
            #start=time.time()
            q.put(np.array(ImageGrab.grab(bbox=(40,180,1000,840))))
            if stop_threads:
                break

            #cv2.imshow('GRAY33',gray)
            #i=i+2
            #print('1')
            #if cv2.waitKey(1) & 0xFF == ord('q'):
             #   print('q')
             #   cv2.destroyAllWindows()
                #p=mp.Process(target=hang)
                #thread.threading.terminate()
                #threadName.exit()
            #end=time.time()
            #sec=start-end;
            #print(1/sec)
            #x.append(screen1);
            #return x;

    #def try11(n):
     #   print('n')
    q=queue.LifoQueue()
    
    t=threading.Thread(target=trial,name='thread1',args=(q,'thread1',),daemon=True)
    t.start()
    #thread.start_new_thread(trial,("Thread-1", ))
    #current_time=time.time()
    label=[]
    percents=[]
    while True:
        #from multiprocessing.pool import ThreadPool
        #pool = ThreadPool(processes=1)
        #async_result = pool.apply_async(trial, ("Thread-1", )) # tuple of args for foo
        # do some other stuff in the main process
        #screen = async_result.get()
        screen1=q.get()
        with q.mutex:
            q.queue.clear()
        screen1=cv2.resize(screen1,(480,320))
        
        #screen1=np.array(ImageGrab.grab(bbox=(40,180,1000,840)))
        screen1=cv2.cvtColor(screen1, cv2.COLOR_BGR2RGB)
        gray=cv2.resize(cv2.cvtColor(screen1,cv2.COLOR_BGR2GRAY),(480,320))
        screen2=cv2.resize(screen1,(480,320))
        cv2.imshow('window',screen2)
        gray=cv2.resize(cv2.cvtColor(screen1,cv2.COLOR_BGR2GRAY),(480,320))
        cv2.imshow('GRAY',gray)
        cv2.imshow('GaussianBlur',cv2.resize(cv2.GaussianBlur(screen1,(45,45),10),(480,320)))
        #img5=cv2.imread(r'C:\Users\Abhi\Desktop\Robotics_course_upenn\Motion_planning\Plant\plantvillage\Tomato___Late_blight\0ab1cab4-a0c9-4323-9a64-cdafa4342a9b___GHLB2 Leaf 8918.JPG')
        labels=predict1.predict1(model_final,screen1)
        #print(a)
        data=pd.DataFrame(columns={'label','percent'})
        label.append(labels)
        percent=attributes.attributes(img5)
        percents.append(percent)
        
        #print "loop took {} seconds",format(t)
        #current_time=time.time()
        #time.sleep(1)
        #print('sss')
        if cv2.waitKey(1) & 0xFF == ord('q'):
                print('q')
                cv2.destroyAllWindows()
                data['label']=label
                data['percent']=percents
                data.to_csv(r'datasheet_very_2nd_useful.csv', index = False, header=True)
                stop_threads=True
                sys.exit()
                break

