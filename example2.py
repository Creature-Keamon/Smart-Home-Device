import tekore as tk #imports tekore

client_id = '' #establishes client id
client_secret = '' #establishes the client secret id

app_token = tk.request_client_token(client_id, client_secret) #collects a client token, used to make basic calls

spotify = tk.Spotify(app_token) #creates spotify object

album = spotify.album('4SD2UxRO9OgeSCQK0PN7cC') #sets album to Sleep Token's "This Place Will Become Your Tomb"
for track in album.tracks.items:
    print(track.track_number, track.name) #prints tracklisting 