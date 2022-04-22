# Leonardo Moracci, Computer Science BEng (Anglia Ruskin University Cambridge)
# Final project
# SID: 1812939


# import necessary packages


#import webcam essential
import cv2
import numpy as np

#import mediapipe and tensorfloww
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

#import bulb library
from PyP100 import PyL530

#import essential text-to-speech libraries
import gtts
from playsound import playsound
import os
#import time
import time
import datetime

from numba import jit, cuda

#import speech to text libraries
import pyaudio
import websockets
import asyncio
import base64
import json
from configure import auth_key


#Get URL and KEY from AssemblyAI
URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
auth_key = "504e14c022ec43bfb965990ffda5841b"



#get the time
current_time = datetime.datetime.now()

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)
 
while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 
  

    if className == 'okay':
        l530 = PyL530.L530("192.168.0.157", "l.moracci@hotmail.it", "Morach92") #Creating a L530 bulb object

        l530.handshake() #Creates the cookies required for further methods
        l530.login() #Sends credentials to the plug and creates AES Key and IV for further methods

        #All the bulbs have the PyP100 functions and additionally allows for setting brightness, colour and white temperature
        l530.setBrightness(100) #Sends the set brightness request
        l530.setColorTemp(2700) #Sets the colour temperature to 2700 Kelvin (Warm White)
        l530.setColor(100, 100) #Sends the set colour request



    if className == 'fist':
        l530 = PyL530.L530("192.168.0.157", "l.moracci@hotmail.it", "Morach92") #Creating a L530 bulb object

        l530.handshake() #Creates the cookies required for further methods
        l530.login() #Sends credentials to the plug and creates AES Key and IV for further methods

        #All the bulbs have the PyP100 functions and additionally allows for setting brightness, colour and white temperature
        l530.setBrightness(1) #Sends the set brightness request
        l530.setColorTemp(2700) #Sets the colour temperature to 2700 Kelvin (Warm White)
        l530.setColor(100, 100) #Sends the set colour request

    
    if className == 'thumbs up':

        #Text to speech from input
        tts = gtts.gTTS("Alexa, turn on the light")
        #Get date and time to save the document 
        date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
        #filename with datestring attached
        filename = "voice"+date_string+".mp3"
        #save the file
        tts.save(filename)
        #play the file
        playsound(filename)
        #delete the file
        os.remove(filename)

    
    if className == 'thumbs down':

        #Text to speech from input
        tts = gtts.gTTS("Alexa, turn off the light")
        #Get date and time to save the document 
        date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
         #filename with datestring attached
        filename = "voice"+date_string+".mp3"
        #save the file
        tts.save(filename)
        #play the file
        playsound(filename)
        #delete the file
        os.remove(filename)


    if className == 'peace':

        #Text to speech from input
        tts = gtts.gTTS("Alexa, What is the weather")
        #Get date and time to save the document 
        date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
         #filename with datestring attached
        filename = "voice"+date_string+".mp3"
        #save the file
        tts.save(filename)
        #play the file
        playsound(filename)
        #delete the file
        os.remove(filename)

    #initialize microphone object

        FRAMES_PER_BUFFER = 3200
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        p = pyaudio.PyAudio()
 
     # starts recording
 
        stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
        ) 
         
         #Get the connection
        async def send_receive():
         print(f'Connecting websocket to url ${URL}')
         async with websockets.connect(
         URL,
         extra_headers=(("Authorization", auth_key),),
         ping_interval=5,
         ping_timeout=20
         ) as _ws:
           await asyncio.sleep(0.1)
           print("Receiving SessionBegins ...")
           session_begins = await _ws.recv()
           print(session_begins)
           print("Sending messages ...")
           async def send():
             while True:
                try:
                   data = stream.read(FRAMES_PER_BUFFER)
                   data = base64.b64encode(data).decode("utf-8")
                   json_data = json.dumps({"audio_data":str(data)})
                   await _ws.send(json_data)
                except websockets.exceptions.ConnectionClosedError as e:
                   print(e)
                   assert e.code == 4008
                   break
                except Exception as e:
                   assert False, "Not a websocket 4008 error"
                await asyncio.sleep(0.01)
          
             return True
       #recive the connection
           async def receive():
                while True:
                 try:
                   result_str = await _ws.recv()
                   print(json.loads(result_str)['text'])
                 except websockets.exceptions.ConnectionClosedError as e:
                   print(e)
                   assert e.code == 4008
                   break
                 except Exception as e:
                   assert False, "Not a websocket 4008 error"
        
         #send_result, receive_result = await asyncio.gather(send(), receive())
           await asyncio.wait(
             [asyncio.create_task(send()), asyncio.create_task(receive())],
             return_when=asyncio.FIRST_COMPLETED
             )

      
        #while loop. 
        #Future implementation might consider stop the loop after a few seconds.
        while True:
          asyncio.run(send_receive())



#If Q is pressed, stop the program.
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()