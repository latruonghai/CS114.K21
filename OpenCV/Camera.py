#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2 # Import thư viện OpenCV\co
import os
import dlib
# Tạo bộ nhận diện khuôn mặt
files = input("Nhap ten: ")
createf = files.replace(".com","")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

sampleNum = 0
while True:
    # Chụp lại từng khung hình
    ret, frame = cap.read()

    # Quá trình nhận diện sẽ được thực hiện trên ảnh xám (Đen/Trắng)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Chuyển ảnh màu sang ảnh xám

    # Thực thi Face Detection
    faces = faceCascade.detectMultiScale(gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Tìm thấy {0} khuôn mặt!".format(len(faces)))

    
    # Vẽ một hình tứ giác xung quanh các khuôn mặt phát hiện được. Vẽ trên ảnh màu.
    for (x, y, w, h) in faces:
        #cv2.putText(frame,"Thanh Ho",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0) ,2, cv2.LINE_AA ),
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        if os.path.exists(createf):
            cv2.imwrite(createf+"/User"+'.'+ str(sampleNum) + ".jpg", frame[y:y+h,x:x+w])
        else:
            os.mkdir(createf)
            cv2.imwrite(createf+"/User"+'.'+ str(sampleNum) + ".jpg", frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Face Detection', frame) # Hiển thị kết quả ra màn hình
    if cv2.waitKey(1) & 0xFF == ord('q'): # Nhấn phím q để dừng
        break
    elif sampleNum>19:
        break

# Trước khi kết thúc chương trình, ta phải giải phóng tài nguyên camera
cap.release()
cv2.destroyAllWindows()


# In[ ]:




