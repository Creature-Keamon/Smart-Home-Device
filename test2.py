import tekore as tk

conf = tk.config_from_environment()
scope = tk.scope.user_top_read
token = tk.prompt_for_user_token(*conf, scope=scope)

spotify = tk.Spotify(token)
artist = spotify.current_user_top_artists(limit=1).items[0]
albums = spotify.artist_albums(artist.id)

print(f'Albums of {artist.name}:')
for a in albums.items:
    print(a.release_date, a.name)