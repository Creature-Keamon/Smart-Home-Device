from chatgpt_wrapper import ApiBackend
import tekore as tk
client_id = 'b09f34e772a444288af5ac9f7628958c'
client_secret = '90732bb485e14c3baa00a02a6fa1fb87'
app_token = tk.request_client_token(client_id, client_secret)

success, response, message = bot.ask("Hello, world!")
if success:
    print(response)
else:
    raise RuntimeError(message)

bot = ApiBackend()

def spotify_function(token, tracks):
    # Call the API
    spotify = tk.Spotify(token)
    album = spotify.album('3RBULTZJ97bvVzZLpxcB0j') # recieves data about this album

    # Use the response
    for track in album.tracks.items:
        tracks = (track.track_number, track.name)
        print(tracks)


                      
        
      if request == "music":
          spotify_function(app_token, tracklist)