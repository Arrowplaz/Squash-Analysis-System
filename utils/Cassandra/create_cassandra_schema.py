from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import uuid

# Configuration
CASSANDRA_HOSTS = ['127.0.0.1']
CASSANDRA_PORT = 9042
CASSANDRA_USERNAME = 'your_username'
CASSANDRA_PASSWORD = 'your_password'

KEYSPACE = 'squash_ai'
TABLE_NAME = 'matches'

def create_schema(session):
    """Create the keyspace and matches table."""
    session.execute(f"""
    CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
    WITH replication = {{ 'class': 'SimpleStrategy', 'replication_factor': '1' }}
    """)

    session.set_keyspace(KEYSPACE)
    session.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        match_id UUID,
        game_id UUID,
        player_1_name TEXT,
        player_2_name TEXT,
        skill_level TEST,
        gender TEXT,
        player_1_nationality TEXT,
        player_2_nationality TEXT,
        detections LIST<FROZEN<tuple<TIMESTAMP, INT, INT>>>,
        PRIMARY KEY (match_id, game_id)
    )
    """)

def add_game(session, match_id, game_data):
    """Insert a new game into the matches table."""
    query = f"""
    INSERT INTO {TABLE_NAME} (
        match_id, game_id, player_1_name, player_2_name,
        player_1_skill_level, player_2_skill_level,
        player_1_gender, player_2_gender,
        player_1_nationality, player_2_nationality,
        detections
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    session.execute(query, (match_id, *game_data))

def main():
    # Connect to Cassandra
    auth_provider = PlainTextAuthProvider(CASSANDRA_USERNAME, CASSANDRA_PASSWORD)
    cluster = Cluster(CASSANDRA_HOSTS, port=CASSANDRA_PORT, auth_provider=auth_provider)
    session = cluster.connect()

    # Create schema
    create_schema(session)

    # Example match and game data
    match_id = uuid.uuid4()
    game_data = (
        uuid.uuid4(), "Player One", "Player Two", 6.0, 5.5,
        "M", "F", "Country A", "Country B",
        [(uuid.uuid4(), 120, 250), (uuid.uuid4(), 320, 450)]  # Example detections
    )
    add_game(session, match_id, game_data)

    print(f"Match {match_id} and Game {game_data[0]} added successfully.")
    session.shutdown()
    cluster.shutdown()

if __name__ == "__main__":
    main()
