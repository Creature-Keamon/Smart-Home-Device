#sets everything up
import speech_recognition as sr
from chatgpt_wrapper import ChatGPT
import pyttsx3
import asyncio
engine = pyttsx3.init()
mic = sr.Microphone()
r = sr.Recognizer()
bot = ChatGPT()

async def getweather(): # defines "get weather"
  import python_weather
  import os #imports weather libraries
  # declare the client
  async with python_weather.Client(format=python_weather.METRIC) as client:

    # fetch a weather forecast from a city
    weather = await client.get("Christchurch") # Christchurch is temporary
  
    # returns the current day's forecast temperature (int)
    print(weather.current.temperature)
  
    # get the weather forecast for a few days
    for forecast in weather.forecasts:
      print(forecast.date, forecast.astronomy)

    if __name__ == "__main__":
      if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

      asyncio.run(getweather())

#sets the active state variable to false
active_state = False


getweather()


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


  except:
      print("I couldn't understand")   #prints this if it can't understand


    #sends chatgpt the request and saves it as a variable
      response = bot.ask(request)
    
    #prints the response from chatGPT
      print(response) 
    
    #makes chatgpt say the output
      engine.say(response)
      engine.runAndWait()

