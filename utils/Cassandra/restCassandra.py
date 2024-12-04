import uuid
import json
from datetime import datetime

class MatchStorage:
    def __init__(self, session, keyspace):
        self.session = session
        self.keyspace = keyspace
        self.table = "matches"

    def create_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table} (
            match_id INT PRIMARY KEY,
            location TEXT,
            skill_level INT,
            player1_name TEXT,
            player2_name TEXT,
            player1_right_handed INT,
            player2_right_handed INT,
            frames LIST<TEXT>
        );
        """
        self.session.execute(query)

    def insert_match(self, location='test', skill_level=0, player1_name='p1', player2_name='p2', 
                     player1_right_handed=0, player2_right_handed=1, video_frames=[], player_detections=[]):
        match_id = 21
        
        # Combine frames and detections into serialized strings
        frames = []
        for frame_number, (frame, player_dict) in enumerate(zip(video_frames, player_detections), start=1):
            print(frame, player_dict)
            frame_data = {
                "frame": frame_number,
                "player_dict": player_dict
            }
            frames.append(json.dumps(frame_data))  # Serialize frame data
        
        query = f"""
        INSERT INTO {self.keyspace}.{self.table} 
        (match_id, location, skill_level, player1_name, player2_name, 
         player1_right_handed, player2_right_handed, frames) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        self.session.execute(query, (
            match_id, location, skill_level, player1_name, player2_name, 
            player1_right_handed, player2_right_handed, frames
        ))

    def get_match_frames(self, match_id):
        """
        Query match_id and return frames as [frame, player_dict] pairs.
        """
        query = f"SELECT frames FROM {self.keyspace}.{self.table} WHERE match_id = %s;"
        result = self.session.execute(query, (match_id,))

        for row in result:
            # Deserialize the JSON strings in frames list
            frames_data = [
                json.loads(frame_data) for frame_data in row.frames
            ]
            return frames_data

        # If no match is found, return None
        return None
