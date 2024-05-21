# mask_detection

This project implements a real-time mask detection system using OpenCV, Keras, and a custom-trained deep learning model. It allows you to collect data for training your own model (using `datacolletor.py`) and then utilize the trained model for real-time mask detection in a video stream (using the application based on the code in the Jupyter Notebook).

# **Data Collection (datacolletor.py)**

1. **Function:** This script helps collect images of faces with and without masks for training your model. It captures video from your webcam, detects faces using the Haar cascade classifier, and saves the detected faces to separate folders based on whether a mask is detected or not.

2. **Running the Script:**
    * Save the script as `datacolletor.py`.
    * Ensure you have OpenCV (`pip install opencv-python`) installed.
    * Run the script from your terminal: `python datacolletor.py`

3. **Customization:**
    * The script creates folders named `images/face_without_mask` and `images/face_with_mask` to store captured images. You can modify these folder names as needed.
    * The script stops collecting images after 500 faces are captured (you can change the value of `count` in the `while` loop).

# **Data Preprocessing and Training (Jupyter Notebook)**

1. **Import Libraries:**
    * The code imports necessary libraries like OpenCV, NumPy, matplotlib, and Keras for image processing, data manipulation, visualization, and deep learning model building, respectively.

2. **Data Path and Preprocessing:**
    * The script defines the path to the folder containing your collected images (`path='images'`).
    * It iterates through subfolders within the `path` directory, assuming each subfolder represents a class (e.g., `face_with_mask`, `face_without_mask`).
    * For each image, it resizes it to a fixed dimension (`imgDimension=(32,32,3)`) and converts it to grayscale for model compatibility.
    * The script also creates a list to store the class labels (0 for mask, 1 for no mask) corresponding to each image.

3. **Data Splitting:**
    * The script splits the collected data into training, testing, and validation sets using `train_test_split` from scikit-learn.
    * The training set is used to train the model, the testing set is used to evaluate the model's performance on unseen data, and the validation set is used to monitor the model's training process and prevent overfitting.
    * You can adjust the `testRatio` and `valRatio` variables to control the proportions of each set.

4. **Data Visualization (Optional):**
    * The script optionally plots a bar chart to visualize the distribution of images across different classes.

5. **Preprocessing Function:**
    * The `preprocessing` function converts the images to grayscale, applies histogram equalization for better contrast, and normalizes pixel values between 0 and 1 for model training.

6. **Reshaping Data:**
    * The script reshapes the image data to add an extra dimension for the batch size, making it suitable for Keras models that expect 4D input (batch, height, width, channels).

7. **Data Augmentation (Optional):**
    * The script defines an `ImageDataGenerator` object from Keras for data augmentation. This technique artificially increases the dataset size by creating variations of existing images (e.g., applying random shifts, zooms, rotations) to improve model robustness.
    * You can enable data augmentation by uncommenting the lines related to `dataGen.fit(x_train)`.

8. **One-Hot Encoding:**
    * The script uses `to_categorical` from Keras to convert class labels (0 or 1) into one-hot encoded vectors, which are a more suitable representation for multi-class classification problems.

9. **Model Definition:**
    * The script defines a Convolutional Neural Network (CNN) architecture using Keras' `Sequential` API.
    * The CNN consists of convolutional layers for feature extraction, pooling layers for dimensionality reduction, activation functions (ReLU) for introducing non-linearity, dropout layers for preventing overfitting, flattening layers for converting 2D feature maps to 1D vectors, and dense layers for classification.
    * You can experiment with different network architectures and hyperparameters (e.g., number of layers, filter sizes, learning rate) to potentially improve model performance.

10. **Model Compilation:**
    * The script compiles the model using the Adam optimizer, categorical cross-entropy loss function (suitable for multi-class classification), and accuracy metric. You can explore other optimizers and loss functions depending on your dataset and task.
   
11. **Model Training:**
  * The script trains the model using the fit_generator method from Keras, which allows for efficient training with data augmentation if enabled.
It iterates through epochs (complete training cycles), feeding batches of training data to the model and updating its weights based on the calculated loss.
  * The validation data is used to monitor the model's performance during training and prevent overfitting.

12. **Model Saving:**
  * Once training is complete, the script saves the trained model to a file named MyTrainingModel.h5. This file can then be used for real-time mask detection in the application.


# **Real-Time Mask Detection Application**

1. **Libraries:**
    * The application code (based on the provided code snippet) imports libraries like OpenCV, NumPy, and Keras for video capture, image processing, and loading the trained model, respectively.

2. **Face Detection:**
    * The script loads the pre-trained Haar cascade classifier for face detection (`facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')`).
    * It captures video from your webcam using `cv2.VideoCapture(0)`.
    * It iterates through frames in the video stream and detects faces using the classifier.

3. **Preprocessing and Prediction:**
    * For each detected face, the script extracts the face region from the frame.
    * It resizes the face image to the model's input size (`(32, 32)` in this example).
    * It applies the same preprocessing steps (grayscale conversion, normalization) as used during training.
    * The script loads the trained model (`model.load_model('MyTrainingModel.h5')`) and uses it to predict whether the face is wearing a mask or not.

4. **Visualization and Output:**
    * The script defines a threshold value (`threshold`) for the predicted probability. If the predicted probability of a class (mask or no mask) exceeds this threshold, the script draws a bounding box around the face and displays the predicted class label on the frame.
    * It displays the processed video frame with the mask detection results.

5. **Exit Condition:**
    * The application continues processing video frames until the 'q' key is pressed, at which point it releases the video capture object and closes all OpenCV windows.

**Running the Real-Time Mask Detection Application:**

1. **Prerequisites:**
    * Ensure you have the required libraries installed (`opencv-python`, `numpy`, `keras`).
    * Make sure your trained model (`MyTrainingModel.h5`) is saved in the same directory as the application script.

2. **Running the Script:**
    * Save the application code as a Python script (e.g., `mask_detection.py`).
    * Run the script from your terminal: `python mask_detection.py`

The application will start processing your webcam video and display real-time mask detection results.

# Output
  
Without Mask

<img width="960" alt="2024-05-21 (8)" src="https://github.com/get-aastha/mask_detection/assets/108509128/ff48eac3-aab3-43be-b772-684ad73a9787">

<img width="960" alt="2024-05-21 (9)" src="https://github.com/get-aastha/mask_detection/assets/108509128/3aa3a490-52f6-484b-98ed-cc31b29e24c6">

With Mask

<img width="960" alt="2024-05-21 (12)" src="https://github.com/get-aastha/mask_detection/assets/108509128/d45abc7c-827d-4770-b777-74da365492fc">

<img width="960" alt="2024-05-21 (14)" src="https://github.com/get-aastha/mask_detection/assets/108509128/27c1a538-2546-40c3-908f-0e1c5cd51ace">

