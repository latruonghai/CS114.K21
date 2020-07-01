#!/usr/bin/env python
# coding: utf-8

# In[1]:


#La Truong Hai - 18520698
# Import thư viện cv2 để detect face

import cv2
# Import thư viện os để dẫn đường dẫn đến thư mục
import os
import dlib


# In[2]:


faceCasCade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)


# In[3]:


# Doc hinh
hai = ['./MTPs/MTP'+str(i)+'.jpg' for i in range(1,188) if i not in [89,152]]
files = input("Nhap ten Folder : ")
createf = files.replace(".com","")
name_file = input("Nhập tên file đi nào: ")
sampleNum = 0
for img1 in hai:
    img = cv2.imread(img1)
    img = cv2.resize(img,(400,500))
    grayImag = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = faceCasCade.detectMultiScale(
            grayImag,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30,30))
    #Vẽ các đường màu xanh lá quanh khuôn mặt
    for (x,y,w,h) in face:
        sampleNum=sampleNum+1
        #Lưu ảnh khuôn mặt vào thư mục có tên(creatình
        if os.path.exists(createf):
            cv2.imwrite(createf+"/"+name_file+'-'+ str(sampleNum) + ".jpg", cv2.resize(img[y:y+h,x:x+w],(216,216)))
        else:
            os.mkdir(createf)
            cv2.imwrite(createf+"/"+name_file+'-'+ str(sampleNum) + ".jpg", cv2.resize(img[y:y+h,x:x+w],(216,216)))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)
    # Hiển thị ra màn hình
        if sampleNum>200:
            break
    cv2.imshow("Face Detection ", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()


# In[ ]:




