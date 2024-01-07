import cv2


image = cv2.imread('image.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 4)

genderProto= "model/gender_deploy.prototxt"
genderModel = "model/gender_net.caffemodel"
ageProto = "model/age_deploy.prototxt"
ageModel = "model/age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-80)', '(80-100)']
genderList = ["Male", "Female"]

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face_roi = image[y:y + h+14, x:x + w+14]
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)    
    genderNet.setInput(blob)
    gender_predictions = genderNet.forward()
    gender = genderList[gender_predictions.argmax()]    
    ageNet.setInput(blob)
    agePreds = ageNet.forward()

    age = ageList[agePreds[0].argmax()]
    cv2.putText(image, f'Gender: {gender}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(image, f'Age: {age} years', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
