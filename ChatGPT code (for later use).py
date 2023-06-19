from chatgpt_wrapper import ApiBackend
bot = ApiBackend()

success, response, message = bot.ask("Hello, world!")
if success:
    print(response)
else:
    raise RuntimeError(message)

#sets everything up

def chat(input,output):
    #sends chatgpt the request and saves it as a variable
    output = bot.ask(input)
        
    #prints the response from chatGPT
    print(output) 

    if "chatgpt" in request:
          chat(request, response)
          engine.say(response)
          engine.runAndWait()