import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from keras.models import load_model

# Load the face detection model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set detection threshold
threshold = 0.90

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Define font for text display
font = cv2.FONT_HERSHEY_COMPLEX

# Load the trained model
model = load_model('MyTrainingModel.h5')


def preprocessing(img):
    """Preprocess the image."""
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0  # Normalize to range [0, 1]
    return img


def get_className(classNo):
    """Map class index to class name."""
    if classNo == 0:
        return "Mask"
    elif classNo == 1:
        return "No Mask"
    else:
        return "Unknown"


while True:
    success, img_original = cap.read()

    if not success:
        break

    # Detect faces in the image
    faces = facedetect.detectMultiScale(img_original, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the face region
        crop_img = img_original[y:y + h, x:x + w]

        # Resize the face to the model's input size
        img = cv2.resize(crop_img, (32, 32))

        # Preprocess the image
        img = preprocessing(img)

        # Reshape the image for model input (add an extra dimension for batch)
        img = img.reshape(1, 32, 32, 1)

        # Make prediction using the model
        prediction = model.predict(img)

        # Get the predicted class index (use np.argmax instead of predict_classes)
        class_index = np.argmax(prediction, axis=1)[0]

        # Get the probability of the predicted class
        probability_value = np.max(prediction)

        # Draw bounding box and label only if probability exceeds threshold
        if probability_value > threshold:
            if class_index == 0:
                color = (0, 255, 0)  # Green for mask
            else:
                color = (50, 50, 255)  # Blue for no mask

            cv2.rectangle(img_original, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img_original, (x, y - 40), (x + w, y), color, -2)
            class_name = get_className(class_index)
            cv2.putText(img_original, class_name, (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Result", img_original)

    # Exit loop if 'q' key is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
