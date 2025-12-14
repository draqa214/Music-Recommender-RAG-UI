"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import library from "@/components/library"
import { useToast } from "@/hooks/useToast"
import { Home, Search, Library, Music, Play, Pause, SkipBack, SkipForward, Volume2, Maximize2, RotateCw } from "lucide-react"

interface Song {
  has_song: boolean
  name?: string
  artist: string
  album: string
  img_url?: string
  uri?: string
  error?: string
}

type playlist = {
  name: string;
  uri: string;
}


const playlists = [ "Daily Mix 1", "Daily Mix 2", "Daily Mix 3", "Daily Mix 4", "Daily Mix 5", "Daily Mix 6"]
const playlists_links = ['spotify:playlist:37i9dQZF1E39mYlMt3Z2tr', 'spotify:playlist:37i9dQZF1E375f14jdPEml', 'spotify:playlist:37i9dQZF1E39NERAq72Lck', 'spotify:playlist:37i9dQZF1E3a1TfoKeV1hI', 'spotify:playlist:37i9dQZF1E38lSgbkeXQwg', 'spotify:playlist:37i9dQZF1E387RBNAs0GZl' ]
export default function SpotifyPrismatic() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [activeNav, setActiveNav] = useState("home")
  const [recentSongs, setRecentSongs] = useState<Song[]>([])
  const [recommendedSongs, setRecommendedSongs] = useState<Song[]>([])
  const [showUnavailable, setShowUnavailable] = useState(false)
  const [userPlaylists, setUserPlaylists] = useState<playlist[]>([])
  const { toast, ToastContainer } = useToast();


  useEffect(() => {
    fetchRecentSongs()
    fetchRecommendedSongs() // Call the recommender API immediately on mount
    fetchPlaylists()
    const interval = setInterval(fetchRecentSongs, 120000) // Refresh every 2 minutes
    return () => clearInterval(interval)
  }, [])

  const fetchRecentSongs = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      const response = await fetch(`${apiUrl}/get-recent-songs/`)
      const data = await response.json()
      
      if (data.status === 'success') {
        setRecentSongs(data.recent_songs)
        toast({
          title: "Recent Songs Updated",
          description: `Loaded ${data.recent_songs.length} recent tracks`,
          variant: "success"
        })
      } else {
        throw new Error(data.error || 'Failed to fetch recent songs')
      }
    } catch (error: any) {
      console.error("Error fetching recent songs:", error)
      toast({
        title: "Error Loading Recent Songs",
        description: error.message || 'Unable to fetch your recent tracks',
        variant: "destructive"
      })
    }
  }


  const fetchRecommendedSongs = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      const response = await fetch(`${apiUrl}/music-recommender-api/`, { method: "POST" })
      const data = await response.json()
      
      if (data.status === 'success') {
        setRecommendedSongs(data.recommended_songs)
        const successfulRecs = data.recommended_songs.filter((song: Song) => song.has_song)
        toast({
          title: "New Recommendations",
          description: `Generated ${successfulRecs.length} personalized recommendations`,
          variant: "success"
        })
      } else {
        throw new Error(data.error || 'Failed to generate recommendations')
      }
    } catch (error: any) {
      console.error("Error fetching recommended songs:", error)
      toast({
        title: "Recommendation Error",
        description: error.message || 'Unable to generate recommendations',
        variant: "destructive"
      })
    }
  }

  const fetchPlaylists = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      const response = await fetch(`${apiUrl}/get-user-playlists/`);
      const data = await response.json();
      
      if (data.status === "success") {
        setUserPlaylists(data.playlists);
      } else {
        throw new Error(data.error || 'Failed to fetch playlists')
      }
    } catch (error: any) {
      console.error("Error fetching playlists:", error);
      toast({
        title: "Playlist Error",
        description: error.message || 'Unable to load playlists',
        variant: "destructive"
      })
    }
  };

  return (
    <div className="flex flex-col h-screen bg-black text-white">
      <div className="flex flex-1 overflow-hidden">
        <nav className="w-60 bg-black p-6 flex flex-col">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500 text-transparent bg-clip-text text-center">Recofy</h1>
          <ul className="space-y-2 flex-1 mt-4">
            {[
              { icon: Home, label: "Home", id: "home", path:"/"},
              { icon: Search, label: "Search", id: "search", path:"/search"},
              { icon: Library, label: "Your Library", id: "library", path:"/library" }
            ].map((item) => (
              <li key={item.id}>
                <Button
                  variant="ghost"
                  className={`w-full flex items-center gap-2 px-4 py-2 text-sm hover:bg-white/10 ${
                    activeNav === item.id ? "bg-white/20" : ""
                  }`}
                  onClick={() => setActiveNav(item.id)}
                >
                  <item.icon className="w-5 h-5" />
                  {item.label}
                </Button>
              </li>
            ))}
          </ul>

          <div className="mt-4">
          <h3 className="text-lg font-semibold text-white-400 mb-2">PLAYLISTS</h3>
          {userPlaylists.length > 0 ? (
            <ul className="space-y-0">
              {userPlaylists.map((playlist, index) => (
                <li key={index}>
                  <a href={playlist.uri} target="_blank" rel="noopener noreferrer">
                    <Button variant="ghost" className="w-full justify-start hover:text-white text-gray-400">
                      {playlist.name}
                    </Button>
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-400">No playlists found</p>
          )}
          </div>
          <div className="mt-auto">
            <h3 className="text-l font-semibold text-white-400 mb-2 ">DAILY MIX</h3>
            <ul className="space-y-0">
              {playlists.map((playlist, index) => (
                <li key={index}>
                  <a 
                    href={playlists_links[index]} 
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                  <Button variant="ghost" className="w-full justify-start hover:text-white text-gray-400">
                    {playlist}
                  </Button>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </nav>
        <main className="flex-1 overflow-y-auto p-8 bg-gray-900">
          <h2 className="text-3xl font-bold mb-6">Recently Played</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6">
            {recentSongs.map((song, index) => (
              <SongCard key={index} song={song} />
            ))}
          </div>

          {recommendedSongs.length > 0 && (
            <>
              <div className="flex justify-between items-center mt-6">
                <h2 className="text-3xl font-bold">Recommended Songs</h2>
                <Button 
                  onClick={fetchRecommendedSongs} 
                  variant="ghost" 
                  className="flex items-center gap-2 text-sm hover:bg-white/10"
                  title="Refresh recommendations"
                >
                  <RotateCw className="w-5 h-5" />
                </Button>
                <span className="absolute left-1/2 -translate-x-1/2 mt-2 px-2 py-1 text-xs text-white bg-gray-800 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                  Update recommended tracks
                </span>
              </div>
              <div className="flex items-center gap-3 mt-4">
                <label className="text-sm">Show Unavailable</label>
                <input
                  type="checkbox"
                  checked={showUnavailable}
                  onChange={() => setShowUnavailable(!showUnavailable)}
                  className="w-5 h-5 cursor-pointer"
                />
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6 mt-4">
                {recommendedSongs.filter(song => showUnavailable || song.has_song).map((song, index) => (
                  <SongCard key={index} song={song} isUnavailable={!song.has_song} />
                ))}
              </div>
            </>
          )}
        </main>
      </div>
      <ToastContainer />
    </div>
  )
}

function SongCard({ song, isUnavailable = false }: { song: Song; isUnavailable?: boolean }) {
  if (isUnavailable || !song.has_song) {
    return (
      <Card className="bg-gray-800 border-2 border-red-500/30 opacity-60">
        <CardContent className="p-4">
          <div className="relative">
            <div className="w-full aspect-square bg-gray-700 rounded-md mb-4 flex items-center justify-center">
              <span className="text-gray-500 text-xs">Unavailable</span>
            </div>
          </div>
          <h3 className="font-semibold truncate text-gray-300">{song.name}</h3>
          <p className="text-sm text-gray-500 truncate">{song.artist}</p>
          {song.error && (
            <p className="text-xs text-red-400 mt-1 truncate">Error: {song.error}</p>
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <a href={song.uri} target="_blank" rel="noopener noreferrer" className="group">
    <Card className="bg-gray-900 border-2 hover:bg-gray-800 transition-colors duration-300 group">
      <CardContent className="p-4">
      <div className="relative">
        <img 
        src={song.img_url || "/placeholder.svg"} 
        alt={song.album} 
        className="w-full aspect-square object-cover rounded-md mb-4" 
        />
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <Button className="bg-green-500 hover:bg-green-600 hover:scale-105 transition w-12 h-12 flex items-center justify-center">
                <Play className="w-8 h-8 text-black" />
              </Button>
            </div>
          </div>
        <h3 className="font-semibold truncate">{song.name}</h3>
        <p className="text-sm text-gray-400 truncate">{song.artist}</p>
      </CardContent>
    </Card>
    </a>
  )
}
