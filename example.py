import spotipy
from spotipy.oauth2 import SpotifyClientCredentials #imports required libraries

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="",
                                                           client_secret="")) #connects to correct account

results = sp.search(q='Sleep Token', limit=5) #selects artists and track limit
for idx, track in enumerate(results['tracks']['items']): #collects the top 5 tracks of selected artist
    print(idx, track['name']) #prints the tracks