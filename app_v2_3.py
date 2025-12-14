import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import asyncio
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import openai
import os
from groq import Groq
from openai import OpenAI
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
import re
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback
from datetime import datetime


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('music_recommender.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv('REDIRECT_URI')
last_fm_redirect_uri = os.getenv('LAST_FM_REDIRECT_URI')
last_fm_api_key = os.getenv('LAST_FM_API_KEY')
last_fm_api_secret = os.getenv('LAST_FM_API_SECRET')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
open_ai_api_key = os.getenv('GROQ_API_KEY')

# Use Hugging Face model for deployment compatibility
model_name = "sentence-transformers/all-MiniLM-L6-v2"
SCOPE = "user-library-read user-top-read user-read-recently-played playlist-read-private"

with open('scaler_2.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# client = OpenAI(
#     api_key=open_ai_api_key,
# )

client = Groq(
    api_key=open_ai_api_key,
)

pc = Pinecone(
        api_key=pinecone_api_key
    )

index_name = 'musicbot-index-2'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1156,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(index_name)


model = SentenceTransformer(model_name)

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                client_secret=client_secret,
                                                redirect_uri=redirect_uri,
                                                scope=SCOPE))

fetched_track_ids = set()

def get_track_details(track):
    """Fetch details for a given track."""
    track_id = track['id']
    # If the track has already been fetched, skip it
    if track_id in fetched_track_ids:
        return None

    # Mark the track as fetched
    fetched_track_ids.add(track_id)

    name = track['name']
    artist = track['artists'][0]['name']
    album = track['album']['name']
    release_date = track['album']['release_date']
    popularity = track['popularity']
    duration_ms = track['duration_ms']
    
    return {
        "id": track_id,
        "name": name,
        "artist": artist,
        "album": album,
        "release_date": release_date,
        "popularity": popularity,
        "duration_ms": duration_ms,
    }



def fetch_recent_tracks():
    """Fetch the 5 most recently played tracks."""
    logger.info("Starting to fetch recent tracks from Spotify")
    recent_tracks_details = []

    try:
        url = "https://api.spotify.com/v1/me/player/recently-played?limit=20"
        token = sp.auth_manager.get_access_token()
        logger.info(f"Got access token: {token['access_token'][:20]}...")
        
        headers = {
            "Authorization": f"Bearer {token['access_token']}"
        }
        logger.info("Making request to Spotify API")
        response = requests.get(url, headers=headers)
    
        logger.info(f"Spotify API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Retrieved {len(data.get('items', []))} recent tracks from Spotify")
            
            for item in data['items']:
                track = item['track']
                #played_at = item['played_at']
                track_details = get_track_details(track)
                if track_details:
                    #track_details["played_at"] = played_at
                    recent_tracks_details.append(track_details)
                    track_id = track['id']
                    # Add the track to the fetched set (to ensure uniqueness)
                    fetched_track_ids.add(track_id)
                else:  
                    track_id = track['id']
                    track_name = track['name']
                    artist_name = track['artists'][0]['name']
                    album_name = track['album']['name']
                    release_date = track['album']['release_date']
                    popularity = track['popularity']
                    duration_ms = track['duration_ms']
                    recent_tracks_details.append({
                            "id": track_id,
                            "name": track_name,
                            "artist": artist_name,
                            "album": album_name,
                            "release_date": release_date,
                            "popularity": popularity,
                            "duration_ms": duration_ms,
                        })
                    
    except Exception as e:
        logger.error(f"Error fetching recent tracks: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e
                    
    logger.info(f"Returning {len(recent_tracks_details)} recent tracks")
    return recent_tracks_details


def fetch_song_info_from_lastfm_by_mbid(track_name, artist_name):
    # Check if both track_name and artist_name are provided
    if not track_name or not artist_name:
        return {"Error": "Track name or artist name is missing."}
    
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.getInfo",
        "api_key": last_fm_api_key,
        "artist": artist_name,
        "track": track_name,
        "format": "json",
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'track' in data:
            track_info = data['track']
            return {
                "Playcount": track_info.get('playcount', "N/A"),
                "Listeners": track_info.get('listeners', "N/A"),
                "TopTags": [tag['name'] for tag in track_info.get('toptags', {}).get('tag', [])],
                "WikiContent": track_info.get('wiki', {}).get('content', "N/A")
            }
        else:
            return {"Error": "Track not found on Last.fm"}
    else:
        return {"Error": f"API request failed with status code {response.status_code}"}
    

def get_enriched_song_data(songs):
    logger.info(f"Enriching data for {len(songs)} songs")
    enriched_data = {}
    
    for idx,song in enumerate(songs,1):
        try:
            song_id = song['id']
            track_name = song.get('name', '')
            artist_name = song.get('artist', '')

            logger.info(f"Processing Song #{idx}: {track_name} by {artist_name}")
        
            # Handle missing track name or artist name
            if not track_name or not artist_name:
                logger.warning(f"Missing track name or artist for song #{idx}")
                enriched_data[song_id] = {**song, "Last.fm Info": {"Error": "Track name or artist name is missing."}}
                continue
            
            # Fetch information from Last.fm for each song using artist's MBID
            logger.info(f"Fetching Last.fm info for: {track_name}")
            lastfm_info = fetch_song_info_from_lastfm_by_mbid(track_name, artist_name)
            
            # Add Last.fm info while keeping the original song data intact
            enriched_data[song_id] = {**song, "Last.fm Info": lastfm_info}

            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Error processing song #{idx} ({track_name}): {str(e)}")
            enriched_data[song_id] = {**song, "Last.fm Info": {"Error": f"Processing error: {str(e)}"}}
        
    return enriched_data


def extract_songs_info(enriched_recent_tracks):
    rows = []

    # Iterate through enriched_tracks
    for song_id, details in enriched_recent_tracks.items():
        # Extract the main song details
        song_info = {
            "Song ID": song_id,
            "Name": details.get("name", "N/A"),
            "Artist": details.get("artist", "N/A"),
            "Album": details.get("album", "N/A"),
            "Release Date": details.get("release_date", "N/A"),
            "Popularity": details.get("popularity", "N/A"),
            "Duration (ms)": details.get("duration_ms", "N/A")
        }
        
        # Check if 'Last.fm Info' exists and extract those details
        last_fm_info = details.get("Last.fm Info", {})
        
        song_info["Playcount"] = last_fm_info.get("Playcount", "N/A")
        song_info["Listeners"] = last_fm_info.get("Listeners", "N/A")
        song_info["TopTags"] = ', '.join(last_fm_info.get("TopTags", []))  # Convert list to comma-separated string
        song_info["WikiContent"] = last_fm_info.get("WikiContent", "N/A")
        
        # Append the song info to rows
        rows.append(song_info)
    df_recent_tracks = pd.DataFrame(rows)
    return df_recent_tracks


def generate_tags(song_name, artist_name, album_name):
    logger.info(f"Generating AI tags for: {song_name} by {artist_name}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in music tagging."},
        {"role": "user", "content": f"""
Generate 5 unique tags for the following song based on its name, artist, and album. The tags should capture aspects like genre, mood, theme, or target audience. Make sure the tags are concise and specific to this song. Do not use any additional text or formatting, only list the tags in plain text.

- Song Name: {song_name}
- Artist: {artist_name}
- Album: {album_name}
        """}
    ]

    try:
        response = client.chat.completions.create(
            # model="gpt-4o-mini",
            model = "llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )

        tags = response.choices[0].message.content.strip().split('\n')
        logger.info(f"Generated tags for {song_name}: {tags}")
        return tags
    except Exception as e:
        logger.error(f"Error generating tags for {song_name}: {e}")
        return ["pop", "music", "song", "audio", "track"]  # Return default tags instead of None
    

def appending_tags_generated_from_openai(recent_fetched_tracks):
    generated_tags_for_recent_songs = []
    for index, row in recent_fetched_tracks.iterrows():
        tags = generate_tags(song_name=row['Name'], artist_name=row['Artist'], album_name=row['Album'])
        if tags:
            #print(f"Tags for {row['Name']} by {row['Artist']}: {tags} \n")
            generated_tags_for_recent_songs.append(tags)
    return generated_tags_for_recent_songs





def create_embedding(text):
    embedding = model.encode(text, convert_to_tensor=True)
    # Move tensor to CPU if it's on MPS device to avoid conversion issues
    if hasattr(embedding, 'device') and 'mps' in str(embedding.device):
        embedding = embedding.cpu()
    return embedding



def embedding_to_array(embedding):
    # Ensure tensor is on CPU before converting to numpy
    if hasattr(embedding, 'device') and 'mps' in str(embedding.device):
        embedding = embedding.cpu()
    return np.array(embedding.detach().numpy())


embedding_columns = ['Artist_embeddings', 'Album_embeddings', 'TopTags_embeddings']
numerical_columns = ['Popularity', 'Duration (ms)', 'Playcount', 'Listeners']

def combine_features(row):
    # Convert embeddings to numpy arrays, ensuring they're on CPU
    artist_emb = row[embedding_columns[0]]
    album_emb = row[embedding_columns[1]]
    tags_emb = row[embedding_columns[2]]
    
    # Convert tensors to numpy arrays if needed
    if hasattr(artist_emb, 'cpu'):
        artist_emb = artist_emb.cpu().detach().numpy()
    elif hasattr(artist_emb, 'detach'):
        artist_emb = artist_emb.detach().numpy()
    
    if hasattr(album_emb, 'cpu'):
        album_emb = album_emb.cpu().detach().numpy()
    elif hasattr(album_emb, 'detach'):
        album_emb = album_emb.detach().numpy()
        
    if hasattr(tags_emb, 'cpu'):
        tags_emb = tags_emb.cpu().detach().numpy()
    elif hasattr(tags_emb, 'detach'):
        tags_emb = tags_emb.detach().numpy()
    
    return np.concatenate([
        artist_emb,
        album_emb, 
        tags_emb,
        row[numerical_columns].values
    ])


def calculate_composite_embedding(embeddings):
    return np.mean(embeddings, axis=0)


def recommend_songs(query_embeddings, played_song_ids ,top_k = 3):
    logger.info(f"Querying Pinecone for {top_k} recommendations, excluding {len(played_song_ids)} played songs")
    
    try:
        query_embeddings = query_embeddings.tolist()
        metadata_filter = {"Song id": {"$nin": played_song_ids}}
        
        logger.info(f"Pinecone query - embedding dimension: {len(query_embeddings)}")
        query_response = index.query(vector=query_embeddings, top_k=top_k, include_metadata=True, filter=metadata_filter)
        
        matches = query_response.get("matches", [])
        logger.info(f"Pinecone returned {len(matches)} matches")
        
        return matches
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
    

def create_rag_prompt(recommended_songs):
    
    song_details = "\n".join([f"Song: {song['metadata']['Song']}, Artist: {song['metadata']['Artist']}, Album: {song['metadata']['Album']}" for song in recommended_songs])

    # Construct the RAG prompt
    prompt = f"""
    Based on the following list of songs, recommend 3 more similar songs. Each song should include its name, artist, and album:
    
    {song_details}
    
    Provide the names of 3 similar songs, with artist and album details.
    """

    return prompt

def get_system_prompt():
    system_prompt = """
    You are a helpful assistant specialized in music recommendation. Given a list of songs, recommend 3 more similar songs. Each song should include its name, artist, and album.

    The recommended songs should be similar to the ones in the list. The output should be in the following format:

    Song: <song_name>   Artist: <artist_name>   Album: <album_name>
    Song: <song_name>   Artist: <artist_name>   Album: <album_name>
    Song: <song_name>   Artist: <artist_name>   Album: <album_name>

    INSTRUCTIONS:
    - Do not include any additional text or formatting.
    - DO NOT USE ** or __ for bold or italics.
    - Only list the songs in plain text as shown above.
"""
    return system_prompt
def rag_recommendations(recommended_songs):
    logger.info(f"Generating RAG recommendations from {len(recommended_songs)} Pinecone matches")
    
    try:
        # Create the RAG prompt based on the context (recommended songs)
        prompt = create_rag_prompt(recommended_songs)
        logger.info(f"Created RAG prompt with {len(prompt)} characters")
        
        messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": prompt}
    ]
        
        # Call OpenAI's API for recommendations
        logger.info("Calling Groq API for RAG recommendations")
        response = client.chat.completions.create(
            # model="gpt-4o-mini",  # You can also use other engines like gpt-3.5-turbo
            model = "llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=150,
            temperature=1
        )

        # Extract the response text
        recommendations = response.choices[0].message.content.strip()
        logger.info(f"RAG recommendations generated: {recommendations[:100]}...")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating recommendations with Groq: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    

def extract_song_details(response):
    # Regular expression pattern to match the song details in the format:
    # Song: <song_name>   Artist: <artist_name>   Album: <album_name>
    pattern1 = r"\*\*Song:\*\* (.*?)\s+\*\*Artist:\*\* (.*?)\s+\*\*Album:\*\* (.*?)\s"
    pattern2 = r"\*\*Song\*\*: (.*?)\s+\*\*Artist\*\*: (.*?)\s+\*\*Album\*\*: (.*?)\s"
    pattern3 = r"\*\*Song: (.*?)\*\*\s+\*\*Artist: (.*?)\*\*\s+\*\*Album: (.*?)\*\*"
    pattern4 = r"Song:\s*(.*?),\s*Artist:\s*(.*?),\s*Album:\s*(.*)"

    # Find all matches
    matches1 = re.findall(pattern1, response)
    matches2 = re.findall(pattern2, response)
    matches3 = re.findall(pattern3, response)
    matches4 = re.findall(pattern4, response)

    matches = matches1 + matches2 + matches3 + matches4
    
    songs_info = []
    for match in matches:
        song, artist, album = match
        songs_info.append({"Song": song, "Artist": artist, "Album": album})
    
    print("Songs_info:", songs_info)
    return songs_info


# async def prepare_upsert_data(recent_fetched_tracks):
#     upsert_data = []

#     for _, entry in recent_fetched_tracks.iterrows():

#         embedding = entry['combined_features']
#         embedding = embedding.tolist()

#         metadata = {
#             "Song id": str(entry["Song ID"]),
#             "Artist": entry["Artist"],
#             "Album": entry["Album"],
#             "Song": str(entry["Name"])
#         }

#         upsert_data.append({"id": str(entry["Song ID"]), "values": embedding, "metadata": metadata})
#     return upsert_data


# async def check_and_upsert_data(recent_fetched_tracks, index=index):
#     upsert_data =await prepare_upsert_data(recent_fetched_tracks)

#     song_ids = [entry["id"] for entry in upsert_data]
#     existing_ids = set()
#     query_results = index.fetch(ids=song_ids)

#     if query_results and "vectors" in query_results:
#         existing_ids = set(query_results["vectors"].keys())
    
#     new_data = [entry for entry in upsert_data if entry["id"] not in existing_ids]

#     if new_data:
#         index.upsert(vectors=new_data)
#         print(f"Upserted {len(new_data)} new entries.")
#     else:
#         print("No new entries to upsert.")


def perform_tasks(recent_tracks_details):
    logger.info(f"Starting perform_tasks with {len(recent_tracks_details)} tracks")
    
    try:
        #recent_tracks_details = fetch_recent_tracks()
        recent_tracks_details = recent_tracks_details[:5]
        logger.info(f"Processing top 5 tracks: {[track.get('name', 'Unknown') for track in recent_tracks_details]}")
        
        logger.info("Step 1: Enriching song data with Last.fm info")
        enriched_recent_tracks = get_enriched_song_data(recent_tracks_details)

        logger.info("Step 2: Extracting song information")
        df_recent_tracks = extract_songs_info(enriched_recent_tracks)
        df_recent_tracks = df_recent_tracks.drop(columns=['WikiContent'])
        recent_fetched_tracks = df_recent_tracks
        logger.info(f"Extracted info for {len(recent_fetched_tracks)} tracks")

        logger.info("Step 3: Generating AI tags")
        generated_tags_for_recent_songs = appending_tags_generated_from_openai(recent_fetched_tracks)
        processed_tags = [
            ', '.join(tag.strip() for tag in tags if isinstance(tag, str)) for tags in generated_tags_for_recent_songs
        ]
        logger.info(f"Generated tags: {processed_tags}")
        recent_fetched_tracks['generated_tags'] = processed_tags


        logger.info("Step 4: Creating embeddings")
        recent_fetched_tracks['Artist'] = recent_fetched_tracks['Artist'].astype(str)
        recent_fetched_tracks['Album'] = recent_fetched_tracks['Album'].astype(str)
        recent_fetched_tracks['TopTags'] = recent_fetched_tracks['TopTags'].astype(str)
        
        logger.info("Creating artist embeddings")
        recent_fetched_tracks['Artist_embeddings'] = recent_fetched_tracks['Artist'].apply(lambda x: create_embedding(x))
        logger.info("Creating album embeddings")
        recent_fetched_tracks['Album_embeddings'] = recent_fetched_tracks['Album'].apply(lambda x: create_embedding(x))
        logger.info("Creating tags embeddings")
        recent_fetched_tracks['TopTags_embeddings'] = recent_fetched_tracks['generated_tags'].apply(lambda x: create_embedding(x))

        logger.info("Step 5: Scaling numerical features")
        numerical_columns = ['Popularity', 'Duration (ms)', 'Playcount', 'Listeners']
        recent_fetched_tracks[numerical_columns] = loaded_scaler.transform(recent_fetched_tracks[numerical_columns])

        logger.info("Step 6: Combining features")
        recent_fetched_tracks['combined_features'] = recent_fetched_tracks.apply(combine_features, axis=1)

        embeddings = recent_fetched_tracks['combined_features'].tolist()
        logger.info(f"Created {len(embeddings)} combined feature vectors")

        # await check_and_upsert_data(recent_fetched_tracks)

        played_song_ids = []
        for song_name in recent_fetched_tracks['Song ID']:
            played_song_ids.append(song_name)
        logger.info(f"Excluding {len(played_song_ids)} already played songs")

        logger.info("Step 7: Calculating composite embedding")
        composite_embedding = calculate_composite_embedding(embeddings=embeddings)

        logger.info("Step 8: Getting recommendations from Pinecone")
        recommended_songs = recommend_songs(composite_embedding, played_song_ids, top_k=3)
        
        if not recommended_songs:
            logger.warning("No recommendations returned from Pinecone")
            return []
        
        logger.info(f"Got {len(recommended_songs)} recommendations from Pinecone")

        logger.info("Step 9: Generating RAG recommendations")
        openai_recommendations = rag_recommendations(recommended_songs)
        
        if not openai_recommendations:
            logger.warning("No recommendations returned from RAG")
            return []

        logger.info("Step 10: Extracting song details from RAG response")
        songs_info = extract_song_details(openai_recommendations)
        logger.info(f"Extracted {len(songs_info)} final recommendations")

        return songs_info
        
    except Exception as e:
        logger.error(f"Error in perform_tasks: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e


def search_song(song_name, artist_name):
    logger.info(f"Searching for song: '{song_name}' by '{artist_name}'")
    
    try:
        token = sp.auth_manager.get_access_token()

        url ='https://api.spotify.com/v1/search'

        headers = {
            "Authorization": "Bearer " + token['access_token']
        }

        query = f'track:{song_name} artist:{artist_name}'
        logger.info(f"Search query: {query}")

        params = {
            'q':query,
            'type':'track',
            'limit':1
        }
        response = requests.get(url, headers=headers, params=params)
        logger.info(f"Spotify search API response status: {response.status_code}")

        if response.status_code == 200:
            results = response.json()
            tracks = results.get("tracks", {}).get("items", [])
            
            if tracks:
                track = tracks[0]
                logger.info(f"Found song: {track['name']} by {track['artists'][0]['name']}")
                return {
                    "has_song": True,
                    "id": track["album"]['id'],
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "album": track["album"]["name"],
                    "preview_url": track["preview_url"],
                    "uri": track["uri"],
                    "img_url": track["album"]["images"][0]["url"] if track["album"]["images"] else None
                }
            else:
                logger.warning(f"No tracks found for: {song_name} by {artist_name}")
                return "No song found."
        else:
            logger.error(f"Spotify search failed with status {response.status_code}: {response.text}")
            return "Failed to search for the song."
            
    except Exception as e:
        logger.error(f"Error searching for song '{song_name}' by '{artist_name}': {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "Failed to search for the song."
    

def extract_track_id(spotify_uri):
    """
    Extracts the track ID from a Spotify URI.

    Args:
    spotify_uri (str): The Spotify URI in the format 'spotify:track:<track_id>'.

    Returns:
    str: The extracted track ID.
    """
    if spotify_uri.startswith("spotify:track:"):
        return spotify_uri.split(":")[2]
    else:
        raise ValueError("Invalid Spotify URI format")
    

def create_playcard(song):
    """
    Displays a small clickable playcard for a song with its details.

    Args:
    song (dict): A dictionary containing song details.
    """
    # Extract details
    name = song.get("name")
    artist = song.get("artist")
    album_image = song.get("img_url")
    #uri = song.get("uri")
    #track_id = extract_track_id(uri)
    spotify_url = song.get("uri")

    st.markdown(
        f"""
<a href="{spotify_url}" target="_blank" style="text-decoration: none;">
    <div style="
        display: flex;
        align-items: center;
        margin: 0.3vw;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 0.5vw;
        background-color: #f9f9f9;
        width: 25vw;
        height: 8vh;
        transition: transform 0.2s;
        text-align: left;
        max-width: 300px;  /* Maximum width for larger screens */
        max-height: 60px;  /* Maximum height for larger screens */
        box-sizing: border-box;  /* Ensures margins are not part of width/height */
    " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
        <img src="{album_image}" alt="Album Image" style="width: 3.5vw; height: 3.5vw; border-radius: 5px; margin-right: 1vw;">
        <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="margin: 0; font-size: 0.9vw; color: #000; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name}</h4>
            <p style="margin: 1px 0 0; font-size: 0.7vw; color: #666; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{artist}</p>
        </div>
    </div>
</a>
""",
unsafe_allow_html=True,
)


def display_playcards(songs):
    """
    Displays multiple playcards for a list of songs in a row with proper spacing.

    Args:
    songs (list): A list of dictionaries, where each dictionary contains song details.
    """
    # Start the container div with flexbox (for vertical layout)
    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;  /* Stack the playcards vertically */
            justify-content: flex-start;  /* Aligns the cards at the top */
            padding: 1vw;  /* Optional: Adds padding inside the container */
        ">
        """,
        unsafe_allow_html=True,
    )

def get_unique_recent_songs(recent_songs):
    """
    Filters unique songs from the recent songs list based on the song ID.

    Args:
    recent_songs (list): List of dictionaries containing song details.

    Returns:
    list: List of dictionaries containing unique songs.
    """
    unique_songs = {}
    for song in recent_songs:
        if song["id"] not in unique_songs:
            unique_songs[song["id"]] = song

    # Get the most recent unique songs
    return list(unique_songs.values())[:5]

@router.get("/get-recent-songs/")
def get_recent_songs():
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info(f"[{request_id}] Starting get_recent_songs request")
    
    try:
        logger.info(f"[{request_id}] Fetching recent tracks")
        recent_songs = fetch_recent_tracks()
        
        logger.info(f"[{request_id}] Getting unique songs from {len(recent_songs)} tracks")
        unique_songs = get_unique_recent_songs(recent_songs)
        
        logger.info(f"[{request_id}] Processing {len(unique_songs)} unique songs")
        songs_data = []
        
        for i, song in enumerate(unique_songs):
            try:
                logger.info(f"[{request_id}] Processing song {i+1}/{len(unique_songs)}: {song['name']}")
                song_info = search_song(song['name'], song['artist'])
                songs_data.append(song_info)
            except Exception as song_error:
                logger.error(f"[{request_id}] Error processing song {song['name']}: {str(song_error)}")
                songs_data.append({
                    "has_song": False,
                    "name": song['name'],
                    "artist": song['artist'],
                    "error": str(song_error)
                })
        
        logger.info(f"[{request_id}] Successfully processed {len(songs_data)} recent songs")
        return {
            "status": "success", 
            "recent_songs": songs_data,
            "request_id": request_id,
            "total_songs": len(songs_data)
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in get_recent_songs: {str(e)}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        return {
            "status": "failed", 
            "error": str(e),
            "request_id": request_id,
            "error_type": type(e).__name__
        }


@router.post("/music-recommender-api/")
def music_recommender():
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info(f"[{request_id}] Starting music recommendation request")
    
    try:
        # Step 1: Fetch recent tracks
        logger.info(f"[{request_id}] Step 1: Fetching recent tracks")
        recent_songs = fetch_recent_tracks()
        logger.info(f"[{request_id}] Fetched {len(recent_songs)} recent songs")
        
        if not recent_songs:
            logger.warning(f"[{request_id}] No recent songs found")
            return {"status": "failed", "error": "No recent songs found"}
        
        # Step 2: Process tasks (main recommendation logic)
        logger.info(f"[{request_id}] Step 2: Processing recommendation tasks")
        songs_info = perform_tasks(recent_songs)
        logger.info(f"[{request_id}] Generated {len(songs_info)} song recommendations")
        
        if not songs_info:
            logger.warning(f"[{request_id}] No recommendations generated")
            return {"status": "failed", "error": "No recommendations could be generated"}
        
        # Step 3: Search for songs on Spotify
        logger.info(f"[{request_id}] Step 3: Searching songs on Spotify")
        recommended_songs = []
        
        for i, song in enumerate(songs_info):
            try:
                logger.info(f"[{request_id}] Processing song {i+1}/{len(songs_info)}: {song['Song']} by {song['Artist']}")
                
                song_info = search_song(song['Song'], song['Artist'])
                
                if song_info != "No song found." and song_info != "Failed to search for the song.":
                    logger.info(f"[{request_id}] Found song on Spotify: {song['Song']}")
                    recommended_songs.append(song_info)
                else:
                    logger.warning(f"[{request_id}] Song not found on Spotify: {song['Song']} by {song['Artist']}")
                    recommended_songs.append({
                        "has_song": False,
                        "name": song["Song"],
                        "artist": song["Artist"],
                        "album": song.get("Album", "Unknown")
                    })
                    
            except Exception as song_error:
                logger.error(f"[{request_id}] Error processing song {song['Song']}: {str(song_error)}")
                logger.error(f"[{request_id}] Song error traceback: {traceback.format_exc()}")
                recommended_songs.append({
                    "has_song": False,
                    "name": song.get("Song", "Unknown"),
                    "artist": song.get("Artist", "Unknown"),
                    "error": str(song_error)
                })
        
        logger.info(f"[{request_id}] Successfully processed {len(recommended_songs)} recommendations")
        logger.info(f"[{request_id}] Request completed successfully")
        
        return {
            "status": "success", 
            "recommended_songs": recommended_songs,
            "request_id": request_id,
            "total_recommendations": len(recommended_songs)
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Critical error in music_recommender: {str(e)}")
        logger.error(f"[{request_id}] Full traceback: {traceback.format_exc()}")
        return {
            "status": "failed", 
            "error": str(e),
            "request_id": request_id,
            "error_type": type(e).__name__
        }
    

@router.get("/get-user-playlists/")
def get_user_playlists():
    playlists = sp.current_user_playlists(limit = 10)
    if "items" not in playlists:
        return {"status": "error", "message": "Failed to fetch playlists"}
    
    # Filter out the "Liked Songs" playlist and select the first 5 remaining playlists
    playlists = [
        {
            "name": p["name"],
            "uri": p["uri"],  # Spotify URI
        }
        for p in playlists["items"]
        if p["name"] != "Liked Songs"  # Exclude "Liked Songs"
    ][:4]  # Select the first 5 playlists
    
    playlists.insert(0, {
        "name": "Liked Songs",
        "uri": "https://open.spotify.com/collection/tracks",  # Spotify Web URL for Liked Songs  # Album cover from recently liked songs
    })
    
    return {"status": "success", "playlists": playlists}

    # playlists = playlists[:4]
# # Streamlit UI
# def app():
#     st.title("Music Recommendation System")
    
#     recent_songs = fetch_recent_tracks()
#     unique_songs = get_unique_recent_songs(recent_songs)
#     st.write("### Recently Played Songs:")
#     songs_data = []
#     for _,song in enumerate(unique_songs):
#         song_info = search_song(song['name'], song['artist'])
#         create_playcard(song_info)

#     songs_info = perform_tasks(recent_songs)
#     print(songs_info)
#     # Display the recommended songs
#     st.write("### Recommended Songs:")
#     for song in songs_info:
#         print(f"**Song**: {song['Song']} - **Artist**: {song['Artist']} - **Album**: {song['Album']}")
#         song_info = search_song(song['Song'], song['Artist'])
#         if song_info is not "No song found.":
#             create_playcard(song_info)
#         else:
#             st.write("Song is not on spotify!!!    :(")
#             st.write(f"**Song**: {song['Song']} - **Artist**: {song['Artist']}")

