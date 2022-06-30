from base64 import b64encode
import pandas as pd
import geocoder
import folium
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import streamlit as st
from ipywidgets import HTML
from PIL import Image
image = Image.open('maps.jpeg')


CLASSES_LIST = ["FIGHT", "NOFIGHT"]
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

#loading the model
def load_model():
    model_v = tf.keras.models.load_model("viol_model.hdf5")
    return model_v

model_v = load_model()



@st.cache(suppress_st_warning=True)

def run_map():
        ip = geocoder.ip("202.51.247.22")
        ip.latlng

        data = {'Camera Name':  ['PGPR', 'Com2', 'i3 Auditorium'],
                'Latitude': [1.291654, 1.294108, 	1.292711],
                'Longitude': [103.780445,103.773765, 103.773765]
                }

        df = pd.DataFrame(data)

        print (df)

        import sys
        sys.setrecursionlimit(10000)
        print(sys.getrecursionlimit())

        x = int(2)
        location = [df['Latitude'][x],df['Longitude'][x]]
        map = folium.Map(location= location,zoom_start=100)
        folium.Marker(location).add_to(map)
        map


@st.cache(suppress_st_warning=True)
def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):

    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:
            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = model_v.predict(np.expand_dims(frames_queue, axis=0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)

    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()

@st.cache(suppress_st_warning=True)
def video():
    import os

    test_videos_directory = 'test_videos'
    os.makedirs(test_videos_directory, exist_ok=True)

    # output_video_file_path = f'{test_videos_directory}-Output-SeqLen{20}.mp4'
    output_video_file_path = 'C:\\Users\testfi.mp4'

    video_file_path = 'C:\\Users'
    # Perform Action Recognition on the Test Video.
    predict_on_video(video_file_path, output_video_file_path, SEQUENCE_LENGTH = 20)

import streamlit as st
import pyrebase
from streamlit_option_menu import option_menu

firebaseConfig = {
  'apiKey': "AIzaSyDa8RyPDwMVj9Tw0XyPi7QSdcOOh2_wkHY",
  'authDomain': "stay-safe-fa2dd.firebaseapp.com",
  'databaseURL': "https://stay-safe-fa2dd-default-rtdb.firebaseio.com",
  'projectId': "stay-safe-fa2dd",
  'storageBucket': "stay-safe-fa2dd.appspot.com",
  'messagingSenderId': "765879193502",
  'appId': "1:765879193502:web:53588b7dd1cb2bcaf4108f",
  'measurementId': "G-Q5VGFLKBXH"
};

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()

output_video_file_path = 'C:\\Users\testfi.mp4'

@st.cache(suppress_st_warning=True)
def play(output_video_file_path):
  mp4 = open(output_video_file_path,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML("""
  <video width=400 controls>
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url)

def main():
    st.sidebar.title("Welcome to Carma")
    file = st.file_uploader("Pick a file")
    if file == None:
        st.text("Please upload a file")
    else:
        predict_on_video(file, output_video_file_path, SEQUENCE_LENGTH=20)
        video()
        play(output_video_file_path)
    choice = st.sidebar.selectbox("Login/Signup", ["Login","Signup"])

    if choice == "Signup":
        email = st.sidebar.text_input("Enter your email address")
        password = st.sidebar.text_input("Enter your password", type="password")
        st.sidebar.text_input("Region/Department")
        Signup = st.sidebar.button("Signup")
        if Signup:
            user = auth.create_user_with_email_and_password(email,password)
            st.balloons()
            st.success("Account Created")

    elif choice == "Login":
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password",type="password")
        Login = st.sidebar.button("Login")
        forgot = st.sidebar.button("Forgot Password")
        if Login:
            user = auth.sign_in_with_email_and_password(email, password)
            st.balloons()
            st.title("Carma")
            # Navigation Pane


        with st.sidebar:
            selected = option_menu(
                menu_title=None,
                options=["Home", "Maps", "Settings"],
                orientation="horizontal"
            )
        if  selected == "Home":
            pass
        if selected == "Maps":
            st.title("Maps")
            st.image(image, caption= None)
            st.write("https://bit.ly/3ud80Ta")
        if selected == "Settings":
            st.title("Settings")




main()