import cv2
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # pass the blob through the network and obtain the face detections
    # net = faceNet
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[] # list of x,y for drawing the bounding box of the face along with the predicted age and gender
    for i in range(detections.shape[2]):
        #extract the confidence associated within the prediction
        confidence=detections[0,0,i,2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            # compute the (x, y)-coordinates of the bounding box
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

#construct argument parse and parse the argument
ap=argparse.ArgumentParser()
ap.add_argument('--file')

args=ap.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.file if args.file else 0)
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg,faceBoxes=highlightFace(faceNet,frame)
    flipImg= cv2.flip(resultImg,1)
    img = cv2.resize(flipImg, (1160, 800))
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        # extract the ROI of the face and then construct a blob from
        # *only* the face ROI
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        # make predictions on the gender and find the gender list with
        # the largest corresponding probability
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        i = genderPreds[0].argmax()
        genderConfidence = "{:.2f}%".format(genderPreds[0][i] * 100)
        gender=genderList[i]
        print(f'Gender: {gender}, Confidence: {genderConfidence}')

        # make predictions on the age and find the age list with
        # the largest corresponding probability
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        i = agePreds[0].argmax()
        ageConfidence = "{:.2f}%".format(agePreds[0][i] * 100)
        age=ageList[i]
        print(f'Age: {age[1:-1]} years, Confidence: {ageConfidence}')



        cv2.putText(img, f'{gender}, {genderConfidence}, {age}, {ageConfidence}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", img)
