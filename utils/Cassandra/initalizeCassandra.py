from cassandra.cluster import Cluster

cluster = Cluster(['127.0.0.1'])  # Replace with your Cassandra instance IP
session = cluster.connect()

create_keyspace_query = """
CREATE KEYSPACE IF NOT EXISTS squash_ai 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
"""
session.execute(create_keyspace_query)
print("Keyspace 'squash_ai' created or already exists.")

session.set_keyspace('squash_ai')

create_table_query = """
CREATE TABLE IF NOT EXISTS game_details (
    game_id UUID PRIMARY KEY,
    player1_name TEXT,
    player2_name TEXT,
    skill_level INT,
    match_location TEXT,
    player1_right_handed INT,
    player2_right_handed INT
);
"""
session.execute(create_table_query)
print("Table 'game_details' created or already exists.")
