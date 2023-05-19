import python_weather
import asyncio
import os #import required libraries

async def getweather():
  # declare the client
  async with python_weather.Client(unit=python_weather.METRIC) as client:
   
    weather = await client.get("Rolleston") # fetch a weather forecast from a city
    print(weather.current.temperature) # returns the current day's forecast temperature
  
if __name__ == "__main__":
  if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
  asyncio.run(getweather()) #run the function