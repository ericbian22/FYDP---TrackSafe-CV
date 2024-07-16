import cv2
import numpy as py


def initialize_cascades():
  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if face_cascade.empty() or eye_cascade.empty():
        raise IOError("Failed to load cascade files. Check the file path or reinstall OpenCV.")
    return face_cascade, eye_cascade

def configure_camera():
    # Start video capture from the camera
    cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_AVFOUNDATION)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def main():
    face_cascade, eye_cascade = initialize_cascades()
    cap = configure_camera()

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the faces and detect eyes within each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Display the frame with detected faces and eyes
        cv2.imshow('Face and Eye Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

