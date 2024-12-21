import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model
import os
import time

# Load the trained 3D CNN model
model = load_model('fight_detection_model  colab (1).h5')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class VideoCaptureThread:
    def __init__(self, src=0):
        self.video = cv2.VideoCapture(src)
        self.ret, self.frame = self.video.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.video.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
            time.sleep(0.01)  # Adjust sleep to control frame rate

    def read(self):
        with self.lock:
            frame = self.frame.copy()
        return self.ret, frame

    def release(self):
        self.running = False
        self.thread.join()
        self.video.release()

def detect_from_webcam():
    # Initialize threaded video capture
    video = VideoCaptureThread()
    
    # Get video properties
    frame_width = int(video.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    sliding_window = []
    max_frames = 64
    frame_skip = 5  # Process every 5th frame to reduce load
    frame_count = 0

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                # Skip frame to reduce workload
                continue

            # Preprocess the frame
            resized_frame = cv2.resize(frame, (224, 224))
            resized_frame = resized_frame / 255.0
            sliding_window.append(resized_frame)

            # Keep only the last max_frames frames
            if len(sliding_window) > max_frames:
                sliding_window.pop(0)

            # Make prediction when we have enough frames
            if len(sliding_window) == max_frames:
                input_frames = np.array(sliding_window)
                input_frames = np.expand_dims(input_frames, axis=0).astype('float32')

                # Make prediction
                prediction = model.predict(input_frames, verbose=0)
                predicted_class = np.argmax(prediction, axis=1)[0]

                # If fight is detected, overlay text
                if predicted_class == 1:
                    text = "FIGHT"
                    font_scale, font_thickness = 1, 2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    
                    text_x = (frame_width - text_size[0]) // 2
                    text_y = frame_height - 30
                    
                    # Draw semi-transparent red background
                    overlay = frame.copy()
                    rect_pad = 10
                    rect_x = text_x - rect_pad
                    rect_y = text_y - text_size[1] - rect_pad
                    rect_width = text_size[0] + 2 * rect_pad
                    rect_height = text_size[1] + 2 * rect_pad
                    
                    cv2.rectangle(overlay, (rect_x, rect_y), 
                                  (rect_x + rect_width, rect_y + rect_height), 
                                  (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    
                    # Draw text
                    cv2.putText(frame, text, (text_x, text_y), font, 
                                font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Webcam Fight Detection', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        video.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        video.release()
        cv2.destroyAllWindows()

# Start detection on the webcam
detect_from_webcam()
