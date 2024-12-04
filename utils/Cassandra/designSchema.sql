CREATE KEYSPACE squash_ai WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE squash_ai;

CREATE TABLE game_details (
    game_id UUID PRIMARY KEY,
    player1_name TEXT,
    player2_name TEXT,
    skill_level INT,
    match_location TEXT,
    player1_right_handed INT,
    player2_right_handed INT
);
