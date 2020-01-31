import numpy as np
from mtcnn import MTCNN
from scipy import signal
import matplotlib.pyplot as plt

import cv2

tmp = []
tmpface = []
cap = cv2.VideoCapture('aayfryxljh.mp4')
detector = MTCNN()
while(cap.isOpened()):
    ret, frame = cap.read()

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x, y, w, h = detector.detect_faces(frame)[0]['box']
    prob = detector.detect_faces(frame)[0]['confidence']

    print('szerokosc',x+w,'wysokosc',y+h,'prob: ',prob)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = frame[y:y+h,x:x+w]


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    im2 = tmp
    im2face = tmpface
    im1 = frame
    im1face = face
    tmp = im1
    tmpface = im1face

    if len(im2) != 0:
        # print(im1,im2)
        # im = im1 - im2
        print(im1face.shape[0],im2face.shape)
        if(im1face.shape[0] > im2face.shape[0] and im1face.shape[1] > im2face.shape[1]):
            corr = cv2.matchTemplate(im1,im2,cv2.TM_CCOEFF_NORMED)
            corr2 = cv2.matchTemplate(im1face,im2face,cv2.TM_CCOEFF_NORMED)
            corr3 = cv2.matchTemplate(im1,im2face,cv2.TM_CCOEFF_NORMED)
        elif(im1face.shape[0] < im2face.shape[0] and im1face.shape[1] < im2face.shape[1]):
            corr = cv2.matchTemplate(im1, im2, cv2.TM_CCOEFF_NORMED)
            corr2 = cv2.matchTemplate(im2face, im1face, cv2.TM_CCOEFF_NORMED)
            corr3 = cv2.matchTemplate(im1, im2face, cv2.TM_CCOEFF_NORMED)
        else:
            im1face = im1face[0:im2face.shape[0],0:im2face.shape[1]]
            corr = cv2.matchTemplate(im1, im2, cv2.TM_CCOEFF_NORMED)
            corr2 = cv2.matchTemplate(im2face, im1face, cv2.TM_CCOEFF_NORMED)
            corr3 = cv2.matchTemplate(im1, im2face, cv2.TM_CCOEFF_NORMED)

        # corr = signal.correlate2d(im1,im2)
        print('cor', corr)
        print('cor',np.max(corr2))
        print('cor', np.max(corr3))
        # plt.plot(im)
        # plt.show()
        cv2.imshow('frame',im1face)

    # cv2.imshow('frame', im1)
    # input('Get input...')

cap.release()
cv2.destroyAllWindows()