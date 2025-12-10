import json
import random
import re
import sqlite3
import requests

import torch
from dialoguekit.platforms import FlaskSocketPlatform
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

class MusicRecommenderAgent(Agent):
    def __init__(self, id: str):
        """Music Recommender Agent.

        This agent manages multiple music playlists by processing user commands.
        The user can create, add to, remove from, view, or clear playlists.
        
        Args:
            id: Agent id.
        """
        super().__init__(id)
        self.suggest_unused_feature_count = 0
        self.suggest_add_song_count = 0
        self.num_user_msg = 0

        # self.model_path = "data/finetuned_bert"
        # self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        # self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

        # Nueva configuración para LLaMA
        # self.model_path = "data/llama-1b_prompt0_60_PreprocesadoES"
        # self.base_model_id = "meta-llama/Llama-3.2-1B-Instruct"

        # Token de Hugging Face
        # hf_token = "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"
        # login(hf_token)
        
        # Cargar tokenizer desde el adaptador (como ya vimos que tienes los archivos)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Cargar modelo base
        # base_model = AutoModelForCausalLM.from_pretrained(self.base_model_id, trust_remote_code=True)

        # Aplicar el adaptador (LoRA, PEFT)
        # self.model = PeftModel.from_pretrained(base_model, self.model_path)

        # Create a dictionary to track which features the user has used
        self.used_features = {
            "song_added": False,
            "song_removed": False,
            "playlist_viewed": False,
            "playlist_cleared": False,
        }

        # Connect to the database
        self.connection = sqlite3.connect('data/music_database.db', check_same_thread=False)
        self.cursor = self.connection.cursor()


    def welcome(self) -> None:
        """Sends the agent's welcome message."""
        utterance = AnnotatedUtterance(
            "Hello, I'm your music assistant. You can manage your playlist by adding or removing songs. Also you can tell me to clear the entire playlist or to view all songs you have.",
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)
        #self.display_help()

    def goodbye(self) -> None:
        """Sends the agent's goodbye message."""
        # Close the connection
        self.connection.close()
        utterance = AnnotatedUtterance(
            "It was nice managing your playlists. Bye!",
            # dialogue_acts=[DialogueAct(intent=self.stop_intent)],
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def display_help(self) -> None:
        """Displays help information to the user."""
        help_message = (
            "You can manage your playlist by adding or removing songs, putting the name between '' and the artist after 'by'. Also you can tell me to clear the entire playlist or to view all songs you have."
        )

        utterance = AnnotatedUtterance(
            help_message,
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def receive_utterance(self, utterance: Utterance) -> None:
        """Gets called each time there is a new user utterance.

        This function processes playlist commands from the user.

        Args:
            utterance: User utterance.
        """
        message = utterance.text.lower()
        self.num_user_msg += 1

        if message == "exit":
            self.goodbye()
            return
        
        if message == "help":
            self.display_help()
            return

        # Detectar intención usando el modelo
        intent = self.predict_intent(message)
        response = ""
        # Check if the agent is waiting for a song selection
        if hasattr(self, 'waiting_for_song_selection') and self.waiting_for_song_selection:
            if message.isdigit():
                selected_index = int(message) - 1
                if 0 <= selected_index < len(self.possible_songs):
                    selected_song = self.possible_songs[selected_index]
                    song_name = selected_song[0]
                    artist_name = selected_song[1]
                    
                    # Add the selected song to the playlist
                    response = self.add_to_playlist(song_name, artist_name)
                    
                    # Clear the waiting flag
                    self.waiting_for_song_selection = False
                    self.possible_songs = []
                    
                else:
                    response = "Invalid selection. Please enter a valid number from the list."

        elif hasattr(self, 'waiting_for_song_selection2') and self.waiting_for_song_selection2:
           
            selected_indices = message.split(",")
            selected_indices = [int(index.strip()) - 1 for index in selected_indices if index.strip().isdigit()]
            
            # Filtrar los índices válidos
            valid_indices = [i for i in selected_indices if 0 <= i < len(self.possible_songs)]
            
            if valid_indices:
                selected_songs = []
                for selected_index in valid_indices:
                    i=0
                    for track in self.possible_songs:
                        if selected_index == i:
                            song_name = track
                            artist_name = self.possible_songs[song_name]
                            
                            # Agregar la canción seleccionada a la lista de reproducción
                            self.add_to_playlist(song_name, artist_name)
                            selected_songs.append((song_name, artist_name))
                            break
                        else:
                            i += 1

                # Limpiar la bandera y la lista de posibles canciones
                self.waiting_for_song_selection2 = False
                self.possible_songs = []

                # Respuesta con las canciones añadidas
                response = "Songs added to the playlist:\n" + "\n".join(
                    [f"{name} by {artist}" for name, artist in selected_songs]
                )
                
            else:
                response = "Invalid selection. Please enter valid numbers from the list."

        # When was album <album> released?
        # elif "when was album" in message:
           # album_name = message.lower().replace("released?", "").replace("released", "").split("when was album")[-1].strip()  
           # album_name = self.normalize_input(album_name)
           # response = self.get_album_release_year(album_name)

        # When was song <song> by <artist> released?
        elif "when was song" in message:
            song_name = message.lower().replace("released?", "").replace("released", "").split("when was song")[-1].strip()

            # Assume the format could be either 'artist: song' or 'song by artist'
            title, artist = None, None  # Initialize both variables
    
            if " by " in song_name:
                # Format: 'song by artist'
                title, artist = song_name.split(" by ", 1)
                song_name = self.normalize_input(song_name)
    
            # Check if both title and artist are present
            if not title or not artist:
                response = "Please specify both the song title and the artist. For example: 'When was song [Song Name] by [Artist Name] released?'"
            else:
                response = self.get_song_release_year(title, artist)
        
        # How many albums has artist <artist> released?
        # elif "how many albums has artist" in message:
            # artist_name = message.lower().replace("released?", "").replace("released", "").split("how many albums has artist")[-1].strip()  
            # response = self.count_artist_albums(artist_name)
        
        # Which album features song <song> by <artist>
        elif "which album features song" in message:
            song_title = message.split("which album features song")[-1].strip()
            song_title = self.normalize_input(song_title)

            # Initialize title and artist as None
            title, artist = None, None

            # Assume the format could be either 'artist: song' or 'song by artist'
            if " by " in song_title:
                # Format: 'song by artist'
                title, artist = song_title.split(" by ", 1)
                title = self.normalize_input(title)

            # Check if both title and artist are present
            if not title or not artist:
                response = "Please specify both the song title and the artist. For example: 'Which album features song [Song Name] by [Artist Name]?'"
            else:
                response = self.find_album_by_song(title, artist)

        
        elif intent == "add":
            self.used_features["song_added"] = True
            
            title, artist = self.extract_song_info(message, intent)

            # Normalize user input
            title = self.normalize_input(title)
            if(artist):
                artist = self.normalize_input(artist)
                
            results = self.search_song_in_db(title, artist)
            if len(results) == 0:
                response = 'No song was found with that title.'
            elif len(results) == 1:
                song = results[0]
                response =  self.add_to_playlist(song[0], song[1])

        
        elif intent == "remove":
            title, artist = self.extract_song_info(message, intent)
            self.used_features["song_removed"] = True
            
           # Normalize user input
            title = self.normalize_input(title)
            artist = self.normalize_input(artist)
            response = self.remove_song_from_playlist(title, artist)

            #response = self.remove_song_from_playlist(title, artist)
        
        elif intent == "view":
            self.used_features["playlist_viewed"] = True
            response = self.view_playlist()
        
        elif intent == "clear":
            self.used_features["playlist_cleared"] = True
            response = self.clear_playlist()

        else:
            response = "Unknown command. You can add to, remove from, view, or clear playlist."

        agent_response = AnnotatedUtterance(
            response,
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(agent_response)

        # Suggest unused features 
        if self.num_user_msg == 4:
            response = self.suggest_unused_feature(self.used_features)
            self.num_user_msg = 0
            if response:
                agent_response = AnnotatedUtterance(
                    response,
                    participant=DialogueParticipant.AGENT,
                )
                self._dialogue_connector.register_agent_utterance(agent_response)

    def search_song_in_playlist(self, title: str, artist: str) -> str:
        """Search for a song in the playlist by title and artist.

        Args:
        title (str): The title of the song.
        artist (str): The name of the artist.

        Returns:
        str: A message indicating whether the song was found and its information.
        """

        query = '''SELECT * FROM Playlist_songs WHERE LOWER(REPLACE(song_name, \"'\", '')) = ? AND LOWER(artist_name) = ?;'''
        self.cursor.execute(query, (title, artist))
        result = self.cursor.fetchone()  # Obtiene la primera fila que coincide

        return result
       
    def search_song_in_db(self, title: str, artist: str = None):
        """Search for a song in the database by title and artist.
        
        Args:
        title (str): The title of the song.
        artist (str): The name of the artist.

        Returns:
        list: A list of songs
        """
        if artist:
            self.cursor.execute("SELECT title, artist FROM Songs WHERE LOWER(REPLACE(title, \"'\", '')) LIKE ? AND LOWER(artist) = ?", ('%' + title + '%', artist))
        else:
            self.cursor.execute("SELECT title, artist FROM Songs WHERE LOWER(REPLACE(title, \"'\", '')) = ?", (title,))
        
        results = self.cursor.fetchall()

        if len(results) > 1:
            response = "There are several versions of this song:\n"
            for i, song in enumerate(results, 1):
                response += f"{i}. {song[0]} by {song[1]}\n"
            response += "Please select the number of the song you'd like to add."
            # Register the response to the dialogue system
            agent_response = AnnotatedUtterance(
                response,
                participant=DialogueParticipant.AGENT,
            )
            self._dialogue_connector.register_agent_utterance(agent_response)
                
            # Set a flag to wait for user input
            self.waiting_for_song_selection = True
            self.possible_songs = results  # Store the song options for later reference

        return results
    
#-------------------------------------ADD - REMOVE - VIEW - CLEAR--------------------------------------------------

    def add_to_playlist(self, title: str, artist: str):
            """
            Adds a song to the Playlist_songs table in the database.

            Args:
            title (str): The title of the song to add.
            artist (str): The name of the artist (optional if only the title is used).

            Returns:
            str: A confirmation message which may also include a suggestion for the user.
            """

            # Add the song to the Playlist_songs table
            self.cursor.execute('''
                INSERT INTO Playlist_songs (song_name, artist_name)
                VALUES (?, ?)
            ''', (title, artist))

            self.connection.commit()  # Save changes in the db

            response = f'Song "{title.title()}" by "{artist.title()}" added to playlist.'
            agent_response = AnnotatedUtterance(
                response,
                participant=DialogueParticipant.AGENT,
            )
            self._dialogue_connector.register_agent_utterance(agent_response)

            if self.suggest_add_song_count == 0:
                self.suggest_add_song_count += 1
                return f'You added "{title}" by "{artist}" to your playlist. Would you like to know more about it? You can ask things like "When was song {title} by {artist} released?".'
            else:
                self.suggest_add_song_count == 0
                return f'You added "{title}" by "{artist}" to your playlist. Would you like to know more about it? You can ask things like "Which album features song {title} by {artist}?".'

        
    def view_playlist(self):
        """Returns the contents of the playlist_songs table."""
        
        self.cursor.execute("SELECT song_name, artist_name FROM Playlist_songs")
        songs = self.cursor.fetchall()
        
        if songs:
            return "Songs in the playlist:\n" + "\n".join([f'{song[0]} by {song[1]}' for song in songs])
        else:
            return "The playlist is empty."

    def remove_song_from_playlist(self, title: str, artist: str = None):
        """
        Remove a song from the Playlist_songs table.

        Args:
        title (str): The title of the song to remove.
        artist (str): The name of the artist (optional if only the title is used).

        Returns:
        str: A confirmation message.
        """

        result = self.search_song_in_playlist(title, artist)
        if result:
            self.cursor.execute("DELETE FROM Playlist_songs WHERE LOWER(REPLACE(song_name, \"'\", '')) = ? AND LOWER(artist_name) = ?", (title, artist))
            self.connection.commit()
            return f'Removed "{result[0]}" from playlist.'
        else:
            return f'Song "{title.title()}" not found in the playlist.'
        
        
    def clear_playlist(self) -> str:
        """Empty the playlist_songs table by removing all songs."""
        self.cursor.execute("DELETE FROM Playlist_songs")
        self.connection.commit()
        return "Cleared all songs from the playlist."
    
#-----------------------------------------------QUESTIONS---------------------------------------------------------------
    
    def get_album_release_year(self, album_name: str):
        self.cursor.execute("SELECT album, year FROM Songs WHERE LOWER(REPLACE(album, \"'\", '')) = ?", (album_name,))
        result = self.cursor.fetchone()
        if result and result[1] != 0:
            return f'The album "{result[0]}" was released in {result[1]}.'
        elif result[1] == 0:
            return f'The release year of the album is unknown'
        else:
            return f'Album "{album_name.title()}" not found.'

    def get_song_release_year(self, song_name: str, artist_name: str):
        self.cursor.execute("SELECT title, year FROM Songs WHERE LOWER(REPLACE(title, \"'\", '')) = ? AND LOWER(artist) = ?", (song_name,artist_name))
        result = self.cursor.fetchone()
        if result and result[1] != 0:
            return f'The song "{result[0]}" was released in {result[1]}.'
        elif result[1] == 0:
            return f'The release year of the song is unknown'
        else:
            return f'Song "{song_name.title()}" not found.'
        
    def count_artist_albums(self, artist_name: str):
        self.cursor.execute("SELECT COUNT(DISTINCT album) FROM Songs WHERE artist = ? COLLATE NOCASE", (artist_name,))
        result = self.cursor.fetchone()
        if result:
            return f'Artist "{artist_name.title()}" has released {result[0]} albums.'
        else:
            return f'Artist "{artist_name.title()}" not found.'
        
    def count_songs_in_album(self, album_name: str):
        self.cursor.execute("SELECT COUNT(*) FROM Songs WHERE LOWER(REPLACE(album, \"'\", '')) = ?", (album_name,))
        result = self.cursor.fetchone()
        if result:
            return f'The album "{album_name.title()}" has {result[0]} songs.'
        else:
            return f'No songs found for the album "{album_name.title()}".'

        
    def find_album_by_song(self, song_title: str, artist_name: str):
        self.cursor.execute("SELECT album, title FROM Songs WHERE LOWER(REPLACE(title, \"'\", '')) = ? AND LOWER(artist) = ?", (song_title,artist_name))
        result = self.cursor.fetchone()
        if result:
            return f'The song "{result[1]}" is featured in the album "{result[0]}".'
        else:
            return f'Song "{song_title.title()}" not found.'
        
#-------------------------------------------------------------------------------------------------------------------------------------------------
    
    def predict_intent(self, message):
        """
        Clasifica la intención del usuario usando un modelo generativo LLaMA.
        Usa el mismo prompt que en la inferencia por lotes.
        """
        # Usamos el mismo prompt que en X_test
        prompt = f"""
        Given the following user input, classify its intent into one of the following categories:
        - **View**: The user wants to see information.
        - **Remove**: The user wants to delete something.
        - **Add**: The user wants to add something.
        - **Clear**: The user wants to clear or reset something.

        Respond with only one of the mentioned categories. Do not add explanations or additional details.

        User request: 

            [{message}] = """.strip()

        try:
            url = "https://tall-ghosts-refuse.loca.lt/predict" # URL del tunel local hacia la API
            response = requests.post(url, json={"prompt": prompt}, timeout=15)
            response.raise_for_status()
            return response.json()["intention"]
        except Exception as e:
            print(f"Error al consultar el modelo remoto: {e}")
            return "view"

    def suggest_unused_feature(self, used_features: dict):
        """
        This function suggests unused features to the user based on their interaction with the system.
        
        Args:
        used_features (dict): A dictionary that keeps track of the features the user has already used.

        Returns:
        str: A suggestion for an unused feature or None if all features have been used.
        """
        suggestions = []
        if not used_features.get("song_added", False):
            suggestions.append("You can add songs to your playlist by typing the song and artist or just the name of the song.")
        if not used_features.get("playlist_viewed", False):
            suggestions.append("Want to see your playlist? Just ask for it!")
        if not used_features.get("song_removed", False):
            suggestions.append("If you'd like to remove a song from your playlist, you can write the song and artist or just the name of the song.")
        if not used_features.get("playlist_cleared", False):
            suggestions.append("Need to start fresh? I can clear your playlist if you want.")
        
        # Introduce only one feature at a time to avoid overwhelming the user
        if suggestions:
            return suggestions[0]
        return None

    def normalize_input(self, text):
        """Normalize the input by converting it to lowercase and removing punctuation."""

        # Convert text to lowercase
        text = text.lower()
        
        # Remove unnecessary punctuation (except for valid cases like apostrophes in titles)
        text = re.sub(r'[^\w\s,.]', '', text)  # Keeps only word characters, spaces, ',' and '.'
        
        return text

    def extract_song_info(self, text, intent):
        if intent in ["add", "remove"]:
            # Buscar título entre comillas
            title_match = re.search(r"'([^']+)'", text)
            title = title_match.group(1).strip() if title_match else None

            # Buscar artista después de 'by'
            artist_match = re.search(r'by\s+(.+)', text, re.IGNORECASE)
            artist = artist_match.group(1).strip() if artist_match else None

            return title, artist

        return None, None

# Inicia la plataforma Flask con tu nuevo agente
platform = FlaskSocketPlatform(MusicRecommenderAgent)
platform.start()
