import tekore as tk

client_id = 'b09f34e772a444288af5ac9f7628958c'
client_secret = '90732bb485e14c3baa00a02a6fa1fb87'
redirect_uri = 'https://example.com/callback'

conf = tk.config_from_environment()
scope = tk.scope.user_top_read + tk.scope.playlist_modify_private
token = tk.prompt_for_user_token(client_id, client_secret, redirect_uri, scope=scope)

spotify = tk.Spotify(token)
top_tracks = spotify.current_user_top_tracks(limit=5).items
top_track_ids = [t.id for t in top_tracks]
recommendations = spotify.recommendations(track_ids=top_track_ids).tracks

user = spotify.current_user()
playlist = spotify.playlist_create(
    user.id,
    'Tekore Recommendations',
    public=False,
    description='Recommendations based on your top tracks <3'
)
uris = [t.uri for t in recommendations]
spotify.playlist_add(playlist.id, uris=uris)