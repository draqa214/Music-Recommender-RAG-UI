import streamlit as st
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import config_auth as config

# # Spotify Developer credentials
# CLIENT_ID = config.CLIENT_ID  # Replace with your Spotify Client ID
# CLIENT_SECRET = config.CLIENT_SECRET  # Replace with your Spotify Client Secret
# REDIRECT_URI = config.REDIRECT_URI  # Replace with your redirect URI
# SCOPE = 'user-library-read playlist-read-private user-top-read'

# # Initialize SpotifyOAuth for authentication
# sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)

# # Function to get the access token
# def get_access_token():
#     token_info = st.session_state.get('token_info')
#     if token_info:
#         access_token = token_info['access_token']
#     else:
#         access_token = None
#     return access_token

# # Function to authenticate user
# def authenticate_user():
#     auth_url = sp_oauth.get_authorize_url()
#     st.markdown(f"Click [here]({auth_url}) to authenticate with Spotify.")

# # Function to fetch and display user data
# def display_user_data():
#     token_info = sp_oauth.get_access_token(st.experimental_get_query_params().get('code', [None])[0])
#     if token_info:
#         st.session_state.token_info = token_info
#         sp = spotipy.Spotify(auth=token_info['access_token'])
#         user = sp.current_user()
#         st.write(f"Hello, {user['display_name']}!")
#         st.write(f"Your Spotify ID: {user['id']}")
#         st.write(f"Your account is linked. Here are your playlists:")
        
#         playlists = sp.current_user_playlists()
#         for playlist in playlists['items']:
#             st.write(f"- {playlist['name']}")
#     else:
#         st.write("Authentication failed. Please try again.")

# # Main page logic
# def main():
#     st.title("Link Your Spotify Account for Personalized Content")
    
#     if 'token_info' in st.session_state:
#         # If the user is already authenticated, display user data
#         display_user_data()
#     else:
#         # If the user is not authenticated, display authentication link
#         authenticate_user()

# if __name__ == '__main__':
#     main()


# Spotify Developer credentials
CLIENT_ID = config.CLIENT_ID  # Replace with your Spotify Client ID
CLIENT_SECRET = config.CLIENT_SECRET  # Replace with your Spotify Client Secret
REDIRECT_URI = config.REDIRECT_URI # Replace with your redirect URI
SCOPE = 'user-library-read playlist-read-private user-top-read'

# Initialize SpotifyOAuth for authentication
sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)

def get_access_token():
    token_info = st.session_state.get('token_info')
    if token_info:
        st.write(f"Access token retrieved from session: {token_info['access_token']}")  # Debug statement
        return token_info['access_token']
    st.write("No access token found in session.")  # Debug statement
    return None

# Function to authenticate user
def authenticate_user():
    auth_url = sp_oauth.get_authorize_url()
    st.markdown(f"Click [here]({auth_url}) to authenticate with Spotify.")
    st.write(f"Authentication URL: {auth_url}")  # Debug statement

def display_user_data():
    # After the user is redirected back with the authorization code
    code = st.experimental_get_query_params().get('code', [None])[0]
    st.write(f"Authorization code received: {code}")  # Debug statement
    
    if code:
        try:
            # Exchange the authorization code for an access token
            token_info = sp_oauth.get_access_token(code)
            st.session_state['token_info'] = token_info  # Save token_info in session state
            
            # Debugging: Log token info to verify it's stored correctly
            st.write(f"Token info saved in session: {st.session_state.get('token_info')}")  # Debug statement

            # Use the access token to create a Spotify object and fetch user data
            sp = spotipy.Spotify(auth=token_info['access_token'])
            user = sp.current_user()

            st.write(f"Hello, {user['display_name']}!")
            st.write(f"Your Spotify ID: {user['id']}")
            st.write(f"Your account is linked. Here are your playlists:")

            playlists = sp.current_user_playlists()
            if playlists['items']:
                st.write(f"Found {len(playlists['items'])} playlists.")  # Debug statement
                for playlist in playlists['items']:
                    st.write(f"- {playlist['name']}")
            else:
                st.write("No playlists found.")  # Debug statement

        except Exception as e:
            st.write(f"Error getting the access token: {str(e)}")
            st.write(f"Error details: {e}")  # Debug statement
    else:
        st.write("Authentication failed. Please try again.")

# Main page logic
def main():
    st.title("Link Your Spotify Account for Personalized Content")

    # Check if the user is already authenticated by checking session state
    if 'token_info' in st.session_state:
        # If the user is authenticated, display their data
        st.write("User is already authenticated. Displaying user data...")  # Debug statement
        display_user_data()
    else:
        # If the user is not authenticated, show the authentication link
        st.write("User is not authenticated. Showing authentication link.")  # Debug statement
        display_user_data()

if __name__ == '__main__':
    main()
