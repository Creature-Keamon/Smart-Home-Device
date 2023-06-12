#imports  libraries
import speech_recognition as sr
import pyttsx3
import asyncio
import python_weather
import tekore as tk
import os
from chatgpt_wrapper import ApiBackend


success, response, message = bot.ask("Hello, world!")
if success:
    print(response)
else:
    raise RuntimeError(message)

#sets everything up
bot = ApiBackend()
engine = pyttsx3.init()
client_id = 'b09f34e772a444288af5ac9f7628958c'
client_secret = '90732bb485e14c3baa00a02a6fa1fb87'
app_token = tk.request_client_token(client_id, client_secret)
mic = sr.Microphone()
r = sr.Recognizer()
location = "Rolleston"
response = "empty"
weatheroutput = "empty"
tracklist = "empty"

async def getweather(weather_script, location): # defines "get weather"

  # declare the client
  async with python_weather.Client(unit=python_weather.METRIC) as client:

    # fetch a weather forecast from a city
    weather = await client.get(location) # Rolleston is temporary
  
    # returns the current day's forecast temperature (int)
    weather_script = "Weather in ", location, " is ", weather.current.temperature, " degrees"

    print (weather_script)
    return weather_script

#sets the active state variable to false
active_state = False

def chat(input,output):
    #sends chatgpt the request and saves it as a variable
    output = bot.ask(input)
        
    #prints the response from chatGPT
    print(output) 


def spotify_function(token, tracks):
    # Call the API
    spotify = tk.Spotify(token)
    album = spotify.album('3RBULTZJ97bvVzZLpxcB0j') # recieves data about this album

    # Use the response
    for track in album.tracks.items:
        tracks = (track.track_number, track.name)
        print(tracks)


#defines the function "passive_listen"
def passive_listen(state): 

  while True: #infinitely loops
      with mic as source: #uses microphone to listen for audio
          print("listening") #for testing purposes only
          r.adjust_for_ambient_noise(source) #adjusts to consider ambient noise
          audio = r.listen(source) #listens
          
      try:
        request = r.recognize_google(audio) #converts to text
        print(request) #prints text
        str(request) #turns text into a string
        request.lower() #makes string lowercase
      
      except:
         print("boohoo") #if it can't understand then it prints boohoo

      if request == "hey frank": #checks if request was "hey frank"
         return state == True #returns "True" if request was hey frank
 
     
active_state = passive_listen(active_state) #calls passive_listen, setting the active state variable's value and passive it's initial value through


if active_state == True:

#activates microphone, informs user when the code is running and adjusts if there is ambient noise       
  with mic as source:
    print("say something: (if you want to end the conversation, say 'end talk')")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    
  try:
    #attempts to convert to text
    request = r.recognize_google(audio)
    print(request)
    str(request)

    if request == "end talk":
        print ("ending talk")    #if the request was end talk, then the code stops

    if request != "end talk":
      #makes AI say the output
      engine.say(request)
      engine.runAndWait()
        
      if request == "weather":
          if __name__ == "__main__":
                if os.name == "nt":
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
                asyncio.run(getweather(weatheroutput, location)) #runs getweather() if the request is weather
              
        
      if request == "music":
          spotify_function(app_token, tracklist)

      if "chatgpt" in request:
          chat(request, response)
          engine.say(response)
          engine.runAndWait()

  except:
    print("I couldn't understand")   #prints this if it can't understand

    

