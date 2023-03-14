#sets everything up
import speech_recognition as sr
from chatgpt_wrapper import ChatGPT
import pyttsx3
engine = pyttsx3.init()
mic = sr.Microphone()
r = sr.Recognizer()
bot = ChatGPT()

#infinitely loops 
while True: 
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

        #if the request was end talk, then the code stops
        if request == "end talk":
            print ("ending talk")
            break

        #sends chatgpt the request and saves it as a variable
        response = bot.ask(request)
        
        #prints the response from chatGPT
        print(response) 
        
        #makes chatgpt say the output
        engine.say(response)
        engine.runAndWait()

    #prints this if it can't understand
    except:
        print("I couldn't understand")

print("goodbye!")

