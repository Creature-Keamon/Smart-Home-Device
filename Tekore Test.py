import tekore as tk

client_id = 'b09f34e772a444288af5ac9f7628958c'
client_secret = '90732bb485e14c3baa00a02a6fa1fb87'
token = tk.request_client_token(client_id, client_secret)

# Call the API
spotify = tk.Spotify(token)
album = spotify.album('1gjugH97doz3HktiEjx2vY?si=gSMjFaPaSQKMORROu8VcsQ')

# Use the response
for track in album.tracks.items:
    print(track.track_number, track.name)

redirect_uri = 'http://192.168.1.79/'

user_token = tk.prompt_for_user_token(client_id, client_secret, redirect_uri, scope=tk.scope.every)

conf = (client_id, client_secret, redirect_uri, user_token.refresh_token)
tk.config_to_file('tekore.cfg', conf)

conf = tk.config_from_file('tekore.cfg', return_refresh=True)
user_token = tk.refresh_user_token(*conf[:2], conf[3])

spotify.playback_start_tracks('2Gt7fjNlx901pPRkvBiNBZ?si=9a9b8c9ca1634041')