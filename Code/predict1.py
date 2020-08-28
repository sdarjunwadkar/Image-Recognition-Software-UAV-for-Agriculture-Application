def predict1(model_final,img):
    ya={0: 'Tomato___Late_blight', 1:'Tomato___Leaf_Mold' ,2 : 'Tomato___Spider_mites Two-spotted_spider_mite', 3:'Tomato___Tomato_mosaic_virus' , 4:'Tomato___healthy' }
    import cv2
    import numpy as np
    img=cv2.resize(img,(224,224))
    img=np.expand_dims(img,axis=0)
    y1=model_final.predict(img)
    #print(y1)
    y1[y1>0.5]=1
    #print(y1)
    for i in range(5):
    #print(y5[i])
        if y1[0,i]==1:
            return ya[i]
