from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
# from utils.config import URI

URI = URI = "mongodb+srv://abhiroopreddy2003:iNyVu0TwLcuwUe88@testcluster.qpi6t.mongodb.net/?retryWrites=true&w=majority&appName=TestCluster"


def get_db():
    """Establishes a connection to the MongoDB database and returns the matches collection."""
    client = MongoClient(URI, server_api=ServerApi('1'), tlsCAFile=certifi.where())
    db = client.get_database("Squash_Data")  
    return db["Honors_Project"] 

def parse_file_name(file_name):
    parts = file_name.split('_')
    
    p1_name = f"{parts[0]} {parts[1]}"  # First two parts form player 1's name
    p2_name = f"{parts[3]} {parts[4]}"  # Fourth and fifth parts form player 2's name
    location = parts[5]  # Sixth part is the location
    game_number = parts[6]  # Seventh part is the game number
    level_of_play = parts[7]  # Eighth part is the level of play

    return {
        "Player 1": p1_name,
        "Player 2": p2_name,
        "Location": location,
        "Game #": game_number,
        "Level of Play": level_of_play
    }


def insert_match(player1, player2, location, game_number, skill_rating, player1_origin_detections, player2_origin_detections,
                 court_detections, player1_transformed_detections, player2_transformed_detections, points):
    """Inserts a new match record into the database."""
    matches = get_db()
    match_data = {
        "player1": player1,
        "player2": player2,
        "location": location,
        "game_number": game_number,
        "skill_rating": skill_rating,
        "player1_origin_detections": player1_origin_detections, 
        "player2_origin_detections": player2_origin_detections,
        "court detections": court_detections,
        "player1_transformed_detections": player1_transformed_detections,
        "player2_transformed_detections": player2_transformed_detections,
        "points": points
    }

    result = matches.insert_one(match_data)
    return result.inserted_id

def get_match_by_id(match_id):
    """Retrieves a match by its unique ID."""
    from bson.objectid import ObjectId
    matches = get_db()
    return matches.find_one({"_id": ObjectId(match_id)})

def get_all_matches():
    """Fetches all matches from the database."""
    matches = get_db()
    return list(matches.find())

def get_matches_by_player(player_name):
    """Fetches all matches where the given player participated."""
    matches = get_db()
    return list(matches.find({"$or": [{"player1": player_name}, {"player2": player_name}]}))

def delete_match(match_id):
    """Deletes a match by its unique ID."""
    from bson.objectid import ObjectId
    matches = get_db()
    return matches.delete_one({"_id": ObjectId(match_id)}).deleted_count



def main():
    """For Testing"""
    dummy_id = insert_match(
        player1="John Doe",
        player2="Jane Smith",
        location="New York Squash Club",
        game_number=1,
        skill_rating="Advanced",
        player1_detections=[1, 2, 3, 4, 6],
        player2_detections=[7, 8, 0]
    )
    print(f"Inserted dummy match with ID: {dummy_id}")

if __name__ == "__main__":
    main()
