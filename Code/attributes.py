def attributes(img):
    
    import os
    import glob
    import numpy as np 
    import pandas as pd  
    from PIL import Image
    import cv2
    # traverse root directory, and list directories as dirs and files as files

    data=pd.DataFrame(columns={'Label','percent'})
    #i =0
    #f=[]

    #x1=[]
    x2=[]
    #i=0


    import os 
    #basepath = os.getcwd()
    #for entry in os.listdir(basepath):
    #    if (os.path.isdir(os.path.join(basepath, entry))):
            
    #        f.append(entry)
    #        i=i+1
            
    #i=0
    #j=0 

    import os 
    #basepath = os.getcwd()
    #images=[]
    #i=1
    #infectedarea=[]
    #Tarea_a=[]
    #perimeter_a=[]
    #percentage=[]
    #for stri in f:
        
        #for entry in os.listdir(stri):
                
                #print(i)
                #x1.append(i)
                #x2.append(entry)
                #filepath=os.path.join(stri,entry)
                #img = cv2.imread(filepath)
                #images.append(img)   
        #i=i+1
                #screen1=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       	    
                #screen2=cv2.resize(screen1,(480,320))
                #cv2.imshow('window',screen2)
    #ax=0
    #del images[4910]
    #del x1[4910]
    #del x2[4910]

    #del images[1391]
    #del x1[1391]
    #del x2[1391]

    #del images[212]
    #del x1[212]
    #del x2[212]
    #for files in images:
        #cv2.imshow('window',files)
        #cv2.waitkey(500)
        #cv2.destroyAllWindows()
    img = cv2.resize(img,(275,183))
    original = img.copy()
    neworiginal = img.copy() 
    #cv2.imshow('window',neworiginal)
    p = 0 
    for a in range(img.shape[0]):
        for j in range(img.shape[1]):
            B = img[a][j][0]
            G = img[a][j][1]
            R = img[a][j][2]
            if (B > 110 and G > 110 and R > 110):
                p += 1
    totalpixels = img.shape[0]*img.shape[1]
    per_white = 100 * p/totalpixels

    if per_white > 10:
        img[a][j] = [200,200,200]
        #cv2.imshow('color change', img)
  
    blur1 = cv2.GaussianBlur(img,(3,3),1)
    #cv2.imshow('blur1', blur1)    
   
   	#mean-shift algo
    newimg = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 10 ,1.0)
   
    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)
    #cv2.imshow('means shift image',img) 
    
    blur = cv2.GaussianBlur(img,(11,11),1)
   
   	#Canny-edge detection
    canny = cv2.Canny(blur, 160, 290)
    #cv2.imshow('canny edge detection', canny)
   	
    canny = cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
   	
   
   	#contour to find leafs
    bordered = cv2.cvtColor(canny,cv2.COLOR_BGR2GRAY)
    _, contours,hierarchy = cv2.findContours(bordered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
    maxC = 0
    for x in range(len(contours)):													
      		if len(contours[x]) > maxC:													
      			maxC = len(contours[x])
      			maxid = x
    #print(maxid)
    if maxid < (len(contours)):
        perimeter= cv2.arcLength(contours[maxid],True)
        Tarea = cv2.contourArea(contours[maxid])
        #perimeter_a.append(perimeter)
        
        cv2.drawContours(neworiginal,contours[maxid],-1,(0,0,255))
   	#print perimeter
    else :
        print(ax)
    
    
    
    
    #cv2.imshow('Contour',neworiginal)
   	#cv2.imwrite('Contour complete leaf.jpg',neworiginal)
   
   
   
   	#Creating rectangular roi around contour
    height, width, _ = canny.shape
    min_x, min_y = width, height
    max_x = max_y = 0
   	#frame = canny.copy()
   
   	# computes the bounding box for the contour, and draws it on the frame,
    for contour in range(len(contours)):
        if maxid<len(contours):
               (x,y,w,h) = cv2.boundingRect(contours[maxid])
               min_x,max_x = min(x, min_x), max(x+w, max_x)
               min_y,max_y = min(y, min_y), max(y+h, max_y)
               if w > 80 and h > 80:
      			#cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)   #we do not draw the rectangle as it interferes with contour later on
                   roi = img[y:y+h , x:x+w]
                   originalroi = original[y:y+h , x:x+w]
      
    if (max_x - min_x > 0 and max_y - min_y > 0):
   		roi = img[min_y:max_y , min_x:max_x]	
   		originalroi = original[min_y:max_y , min_x:max_x]
   		#cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)   #we do not draw the rectangle as it interferes with contour
   
    #cv2.imshow('ROI', roi)
    img = roi
   
   
   	#Changing colour-space
   	#imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imghls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
   	#cv2.imshow('unfiltered HLS', imghls)
    imghls[np.where((imghls==[30,200,2]).all(axis=2))] = [0,200,0]
    #cv2.imshow('HLS', imghls)
   
   	#Only hue channel
    huehls = imghls[:,:,0]
   	#cv2.imshow('img_hue hls',huehls)
   	#ret, huehls = cv2.threshold(huehls,2,255,cv2.THRESH_BINARY)
   
    huehls[np.where(huehls==[0])] = [35]
    #cv2.imshow('processed_img hue hls',huehls)
   
   
   	#Thresholding on hue image
    ret, thresh = cv2.threshold(huehls,28,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow('thresh', thresh)
   
   
   	#Masking thresholded image from original image
    mask = cv2.bitwise_and(originalroi,originalroi,mask = thresh)
    #cv2.imshow('masked out img',mask)
  
    _, contours,heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
    Infarea = 0
       
    for x in range(len(contours)):
        if x<len(contours):
          		cv2.drawContours(originalroi,contours[x],-1,(0,0,255))
          		#cv2.imshow('Contour masked',originalroi)
          
          		#Calculating area of infected region
          		Infarea += cv2.contourArea(contours[x])
           
    #infectedarea.append(Infarea)
    if Infarea > Tarea:
        Tarea = roi.shape[0]*roi.shape[1]
    #Tarea_a.append(Tarea)
    #per = 100 * Infarea/Tarea
    #percentage.append(per)
        #cv2.waitKey(50)
        #cv2.destroyAllWindows()
    #ax=ax+1     





      #  x1=x1[:-3]
       # x2=x2[:-3]
        #infectedarea=infectedarea[:-3]
    percent=100*Infarea/Tarea
    #data['Label']=label
        #data['Name']=x2
        #data['infectedarea']=Infarea
        #data['Tarea']=Tarea
    return percent
        
        
    
