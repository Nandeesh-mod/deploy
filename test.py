import cv2
import streamlit as st
import numpy as np
import tempfile
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import time
import random

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
score=0
 
x_enemy=random.randint(50,600)
y_enemy=random.randint(50,400)

def enemy():
  global score,x_enemy,y_enemy
  #x_enemy=random.randint(50,600)
  #y_enemy=random.randint(50,400)
  cv2.circle(image, (x_enemy,y_enemy), 25, (0, 200, 0), 5)
  #score=score+1

# Use this line to capture video from the webcam
cap = cv2.VideoCapture(0)


# Set the title for the Streamlit app
st.title("Video Capture with OpenCV")

frame_placeholder = st.empty()

# Add a "Stop" button and store its state in a variable
stop_button_pressed = st.button("Stop")

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()

        if not ret:
            st.write("The video capture has ended.")
            break

        # You can process the frame here if needed
        # e.g., apply filters, transformations, or object detection

        # Convert the frame from BGR to RGB format
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
        image = cv2.flip(image, 1)
         
        imageHeight, imageWidth, _ = image.shape
 
        results = hands.process(image)
   
 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
        font=cv2.FONT_HERSHEY_SIMPLEX
        color=(255,0,255)
        text=cv2.putText(image,"Score",(480,30),font,1,color,4,cv2.LINE_AA)
        text=cv2.putText(image,str(score),(590,30),font,1,color,4,cv2.LINE_AA)
 
        enemy()
 
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
 
 
        if results.multi_hand_landmarks != None:
          for handLandmarks in results.multi_hand_landmarks:
            for point in mp_hands.HandLandmark:
 
    
                normalizedLandmark = handLandmarks.landmark[point]
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
    
                point=str(point)
                #print(point)
                if point=='HandLandmark.INDEX_FINGER_TIP':
                 try:
                     cv2.circle(image, (pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1]), 25, (0, 200, 0), 5)
                     #print(pixelCoordinatesLandmark[1])
                     if pixelCoordinatesLandmark[0]==x_enemy or pixelCoordinatesLandmark[0]==x_enemy+10 or pixelCoordinatesLandmark[0]==x_enemy-10:
                        #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                      #if pixelCoordinatesLandmark[1]==y_enemy or pixelCoordinatesLandmark[1]==y_enemy+10 or pixelCoordinatesLandmark[1]==y_enemy-10:
                        print("found")
                        x_enemy=random.randint(50,600)
                        y_enemy=random.randint(50,400)
                        score=score+1
                        font=cv2.FONT_HERSHEY_SIMPLEX
                        color=(255,0,255)
                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                        enemy()
                 except:
                  pass
        
        cv2.imshow('Hand Tracking', image)
        #time.sleep(1)
 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(score)
            break



        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Display the frame using Streamlit's st.image
        frame_placeholder.image(image, channels="RGB")

        # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
            break

    cap.release()
    cv2.destroyAllWindows()