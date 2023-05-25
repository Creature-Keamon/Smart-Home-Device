import tekore as tk

client_id = 'b09f34e772a444288af5ac9f7628958c'
client_secret = '90732bb485e14c3baa00a02a6fa1fb87'
token = tk.request_client_token(client_id, client_secret)

# Call the API
spotify = tk.Spotify(token)
album = spotify.album('3RBULTZJ97bvVzZLpxcB0j')

# Use the response
for track in album.tracks.items:
    print(track.track_number, track.name)

redirect_uri = 'https://sites.google.com/rollestoncollege.nz/shawnhub-org/home'

user_token = tk.prompt_for_user_token(
    client_id,
    client_secret,
    redirect_uri,
    scope=tk.scope.every
)

conf = tk.config_from_file('tekore.cfg', return_refresh=True)
user_token = tk.refresh_user_token(*conf[:2], conf[3])

conf = (client_id, client_secret, redirect_uri, user_token.refresh_token)
tk.config_to_file('tekore.cfg', conf)

spotify.playback_start_tracks('2Gt7fjNlx901pPRkvBiNBZ?si=9a9b8c9ca1634041')