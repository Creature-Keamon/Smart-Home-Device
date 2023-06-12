from chatgpt_wrapper import ApiBackend
bot = ApiBackend()

success, response, message = bot.ask("Hello, world!")
if success:
    print(response)
else:
    raise RuntimeError(message)

#sets everything up
