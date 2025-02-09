# import streamlit as st

# # Function to create a small clickable playcard for a song
# def create_playcard(song):
#     """
#     Displays a small clickable playcard for a song with its details.

#     Args:
#     song (dict): A dictionary containing song details.
#     """
#     # Extract details
#     name = song.get("name")
#     artist = song.get("artist")
#     album_image = song.get("album_image")
#     uri = song.get("uri")
#     track_id = extract_track_id(uri)
#     spotify_url = f"https://open.spotify.com/track/{track_id}"

#     st.markdown(
# #         f"""
# # <a href="{spotify_url}" target="_blank" style="text-decoration: none;">
# #     <div style="
# #         display: flex;
# #         align-items: center;
# #         margin: 1vw;
# #         border: 1px solid #ddd;
# #         border-radius: 10px;
# #         padding: 0.5vw;
# #         background-color: #f9f9f9;
# #         width: 20vw;
# #         height: 8vh;
# #         transition: transform 0.2s;
# #         text-align: left;
# #         max-width: 220px;  /* Maximum width for larger screens */
# #         max-height: 60px;  /* Maximum height for larger screens */
# #     " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
# #         <img src="{album_image}" alt="Album Image" style="width: 3.5vw; height: 3.5vw; border-radius: 5px; margin-right: 1vw;">
# #         <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
# #             <h4 style="margin: 0; font-size: 0.9vw; color: #000; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name}</h4>
# #             <p style="margin: 1px 0 0; font-size: 0.7vw; color: #666; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{artist}</p>
# #         </div>
# #     </div>
# # </a>
# # """,
# f"""
# <a href="{spotify_url}" target="_blank" style="text-decoration: none;">
#     <div style="
#         display: flex;
#         align-items: center;
#         margin: 4vw;  /* Space between playcards */
#         border: 1px solid #ddd;
#         border-radius: 10px;
#         padding: 0.5vw;
#         background-color: #f9f9f9;
#         width: 20vw;
#         height: 8vh;
#         transition: transform 0.2s;
#         text-align: left;
#         max-width: 220px;  /* Maximum width for larger screens */
#         max-height: 60px;  /* Maximum height for larger screens */
#         box-sizing: border-box;  /* Ensures margins are not part of width/height */
#     " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
#         <img src="{album_image}" alt="Album Image" style="width: 3.5vw; height: 3.5vw; border-radius: 5px; margin-right: 1vw;">
#         <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
#             <h4 style="margin: 0; font-size: 0.9vw; color: #000; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name}</h4>
#             <p style="margin: 1px 0 0; font-size: 0.7vw; color: #666; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{artist}</p>
#         </div>
#     </div>
# </a>
# """,
# unsafe_allow_html=True,
# )
    

# def display_playcards(songs):
#     """
#     Displays multiple playcards for a list of songs in a row with proper spacing.

#     Args:
#     songs (list): A list of dictionaries, where each dictionary contains song details.
#     """
#     # Start the container div with flexbox
#     st.markdown(
#         """
# <div style="
#     display: flex;
#     flex-direction: column;  /* Stack the playcards vertically */
#     gap: 2vh;  /* Adds vertical spacing between playcards */
#     justify-content: flex-start;  /* Aligns the cards at the top */
#     padding: 1vw;  /* Optional: Adds padding inside the container */
# ">
#         """,
#         unsafe_allow_html=True,
#     )

#     # Generate playcards for each song
#     create_playcard(song)

#     # Close the container div
#     st.markdown("</div>", unsafe_allow_html=True)



# # Function to extract the track ID
# def extract_track_id(spotify_uri):
#     if spotify_uri.startswith("spotify:track:"):
#         return spotify_uri.split(":")[2]
#     else:
#         raise ValueError("Invalid Spotify URI format")

# # Mock recommended songs
# recommended_songs = [
#     {
#         "name": "Blinding Lights",
#         "artist": "The Weeknd",
#         "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
#         "uri": "spotify:track:0VjIjW4GlUZAMYd2vXMi3b",
#     },
#     {
#         "name": "Levitating",
#         "artist": "Dua Lipa",
#         "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
#         "uri": "spotify:track:463CkQjx2J3Dmx3q817MZt",
#         "img_url" : ""
#     },
#     {
#         "name": "Watermelon Sugar",
#         "artist": "Harry Styles",
#         "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
#         "uri": "spotify:track:6UelLqGlWMcVH1E5c4H7lY",
#     },
#     {
#        "name": "Watermelon Sugar",
#         "artist": "Harry Styles",
#         "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
#         "uri": "spotify:track:6UelLqGlWMcVH1E5c4H7lY",
#     },
#     {
#        "name": "Watermelon Sugar",
#         "artist": "Harry Styles",
#         "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
#         "uri": "spotify:track:6UelLqGlWMcVH1E5c4H7lY",
#     }
# ]

# # Streamlit App
# st.title("Recommended Songs")

# # Create columns for horizontal layout
# columns = st.columns(len(recommended_songs))
# for i, song in enumerate(recommended_songs):
#     with columns[i]:
#         display_playcards([song])

import streamlit as st

# Function to create a small clickable playcard for a song
def create_playcard(song):
    """
    Displays a small clickable playcard for a song with its details.

    Args:
    song (dict): A dictionary containing song details.
    """
    # Extract details
    name = song.get("name")
    artist = song.get("artist")
    album_image = song.get("album_image")
    uri = song.get("uri")
    track_id = extract_track_id(uri)
    spotify_url = f"https://open.spotify.com/track/{track_id}"

    st.markdown(
        f"""
        <a href="{spotify_url}" target="_blank" style="text-decoration: none;">
            <div style="
                display: flex;
                align-items: center;
                margin: 0.3vw;  /* Space between playcards */
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 0.5vw;
                background-color: #f0f0f0 !important;
                width: 20vw;
                height: 8vh;    
                transition: transform 0.2s;
                text-align: left;
                max-width: 220px;  /* Maximum width for larger screens */
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

    # Loop through the list of songs and display each playcard
    for song in songs:
        create_playcard(song)

    # Close the container div
    st.markdown("</div>", unsafe_allow_html=True)


# Function to extract the track ID
def extract_track_id(spotify_uri):
    if spotify_uri.startswith("spotify:track:"):
        return spotify_uri.split(":")[2]
    else:
        raise ValueError("Invalid Spotify URI format")


# Mock recommended songs
recommended_songs = [
    {
        "name": "Blinding Lights",
        "artist": "The Weeknd",
        "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
        "uri": "spotify:track:0VjIjW4GlUZAMYd2vXMi3b",
    },
    {
        "name": "Levitating",
        "artist": "Dua Lipa",
        "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
        "uri": "spotify:track:463CkQjx2J3Dmx3q817MZt",
    },
    {
        "name": "Watermelon Sugar",
        "artist": "Harry Styles",
        "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
        "uri": "spotify:track:6UelLqGlWMcVH1E5c4H7lY",
    },
    {
        "name": "Watermelon Sugar",
        "artist": "Harry Styles",
        "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
        "uri": "spotify:track:6UelLqGlWMcVH1E5c4H7lY",
    },
    {
        "name": "Watermelon Sugar",
        "artist": "Harry Styles",
        "album_image": "https://i.scdn.co/image/ab67616d0000b2732c0448b550da2511854404d9",
        "uri": "spotify:track:6UelLqGlWMcVH1E5c4H7lY",
    }
]

# Streamlit App
st.title("Recommended Songs")

# Call function to display playcards
display_playcards(recommended_songs)
