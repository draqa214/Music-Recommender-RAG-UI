"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

type Playlist = {
  name: string;
  uri: string;
};

const playlists = [
  "Daily Mix 1",
  "Daily Mix 2",
  "Daily Mix 3",
  "Daily Mix 4",
  "Daily Mix 5",
  "Daily Mix 6",
];

const playlists_links = [
  "spotify:playlist:37i9dQZF1E39mYlMt3Z2tr",
  "spotify:playlist:37i9dQZF1E375f14jdPEml",
  "spotify:playlist:37i9dQZF1E39NERAq72Lck",
  "spotify:playlist:37i9dQZF1E3a1TfoKeV1hI",
  "spotify:playlist:37i9dQZF1E38lSgbkeXQwg",
  "spotify:playlist:37i9dQZF1E387RBNAs0GZl",
];

export default function Library() {
  const [userPlaylists, setUserPlaylists] = useState<Playlist[]>([]);

  useEffect(() => {
    fetchPlaylists();
    const interval = setInterval(fetchPlaylists, 120000); // Refresh every 2 minutes
    return () => clearInterval(interval);
  }, []);

  const fetchPlaylists = async () => {
    try {
      const response = await fetch("http://localhost:8000/get-user-playlists/");
      const data = await response.json();
      if (data.status === "success") {
        setUserPlaylists(data.playlists);
      }
    } catch (error) {
      console.error("Error fetching playlists:", error);
    }
  };

  return (
    <div className="mt-4 p-4 bg-gray-900 rounded-lg">
      <h3 className="text-lg font-semibold text-gray-400 mb-2">PLAYLISTS</h3>
      {userPlaylists.length > 0 ? (
        <ul className="space-y-2">
          {userPlaylists.map((playlist, index) => (
            <li key={index}>
              <a href={playlist.uri} target="_blank" rel="noopener noreferrer">
                <Button
                  variant="ghost"
                  className="w-full justify-start hover:text-white text-gray-400"
                >
                  {playlist.name}
                </Button>
              </a>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-gray-400">No playlists found</p>
      )}

      <h3 className="text-lg font-semibold text-gray-400 mt-4 mb-2">DAILY MIX</h3>
      <ul className="space-y-2">
        {playlists.map((playlist, index) => (
          <li key={index}>
            <a href={playlists_links[index]} target="_blank" rel="noopener noreferrer">
              <Button
                variant="ghost"
                className="w-full justify-start hover:text-white text-gray-400"
              >
                {playlist}
              </Button>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
