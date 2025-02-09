from flask import Flask, redirect, url_for, session, request, render_template
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from collections import defaultdict
from datetime import datetime, timedelta
import os
import config
from flask import jsonify
import requests
import datetime

# Flask App Configuration
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_COOKIE_NAME'] = 'Spotify_Dashboard'

# Spotify API Configuration
SPOTIFY_CLIENT_ID = config.CLIENT_ID
SPOTIFY_CLIENT_SECRET = config.CLIENT_SECRET
SPOTIFY_REDIRECT_URI = config.REDIRECT_URI

scope = "user-read-recently-played user-top-read playlist-read-private"

AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_BASE_URL = 'https://api.spotify.com/v1/'

@app.route('/')
def index():
    return "Login page <a href='/login'>Login </a>"

@app.route('/login')
def login():
    auth_url = f"{AUTH_URL}?client_id={SPOTIFY_CLIENT_ID}&response_type=code&redirect_uri={SPOTIFY_REDIRECT_URI}&scope={scope}"
    return redirect(auth_url)

@app.route('/api/playtime-stats')
def get_playtime_stats():
    if 'access_token' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    if datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token')

    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }

    # Get recently played tracks
    # We'll fetch the maximum allowed limit of 50 tracks
    response = requests.get(
        API_BASE_URL + 'me/player/recently-played?limit=50',
        headers=headers
    )
    
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch data"}), response.status_code

    recent_tracks = response.json()
    
    # Process the data to get daily playtime
    daily_playtime = defaultdict(int)
    
    for item in recent_tracks.get('items', []):
        # Convert timestamp to date
        played_at = datetime.strptime(item['played_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
        date_str = played_at.strftime('%Y-%m-%d')
        
        # Add track duration (in minutes)
        duration_ms = item['track']['duration_ms']
        daily_playtime[date_str] += duration_ms / (1000 * 60)  # Convert ms to minutes

    # Get the last 7 days
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
    
    # Format data for the graph
    graph_data = [
        {
            'date': date,
            'playtime': round(daily_playtime[date], 2)
        }
        for date in dates
    ]

    return jsonify(graph_data)


@app.route('/callback')
def callback():
    if 'error' in request.args:
        return jsonify({"error": request.args['error']})
    
    if 'code' in request.args:
        req_body = {
            'code': request.args['code'],
            'grant_type': 'authorization_code',
            'redirect_uri': SPOTIFY_REDIRECT_URI,
            'client_id': SPOTIFY_CLIENT_ID,
            'client_secret': SPOTIFY_CLIENT_SECRET
        }

        response = requests.post(TOKEN_URL, data=req_body)
        token_info = response.json()

        session['access_token'] = token_info['access_token']
        session['refresh_token'] = token_info['refresh_token']
        session['expires_at'] = datetime.datetime.now().timestamp() + token_info['expires_in']
        return redirect('/dashboard')
    
@app.route('/playlists')
def get_playlists():
    if 'access_token' not in session:
        return redirect('/login')

    if datetime.datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token')

    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }

    response = requests.get(API_BASE_URL + 'me/playlists', headers=headers)
    playlists = response.json()

    return jsonify(playlists)

@app.route('/refresh-token')
def refresh_token():
    if 'refresh_token' not in session:
        return redirect('/login')

    if datetime.datetime.now().timestamp() > session['expires_at']:
        req_body = {
            'grant_type': 'refresh_token',
            'refresh_token': session['refresh_token'],
            'client_id': SPOTIFY_CLIENT_ID,
            'client_secret': SPOTIFY_CLIENT_SECRET
        }

        response = requests.post(TOKEN_URL, data=req_body)
        new_token_info = response.json()

        session['access_token'] = new_token_info['access_token']
        session['expires_at'] = datetime.datetime.now().timestamp() + new_token_info['expires_in']

        return redirect('/playlists')

@app.route('/dashboard')
def dashboard():
    if 'access_token' not in session:
        return redirect('/login')

    if datetime.datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token')

    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }

    # Get user profile
    user_profile = requests.get(API_BASE_URL + 'me', headers=headers).json()

    # Get user's playlists
    playlists = requests.get(API_BASE_URL + 'me/playlists', headers=headers).json()

    # Get recently played tracks
    recent_tracks = requests.get(
        API_BASE_URL + 'me/player/recently-played?limit=10',
        headers=headers
    ).json()

    # print("Recent Tracks Response:", recent_tracks)

    # Get user's top artists
    top_artists = requests.get(
        API_BASE_URL + 'me/top/artists?limit=10&time_range=short_term',
        headers=headers
    ).json()

    # print("Top Artists Response:", top_artists)

    return render_template(
        'dashboard.html',
        user=user_profile,
        playlists=playlists['items'] if 'items' in playlists else [],
        recent_tracks=recent_tracks['items'] if 'items' in recent_tracks else [],
        top_artists=top_artists['items'] if 'items' in top_artists else []
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8888, debug=True)