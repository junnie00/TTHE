"""A tiny concert/singer database so the whole loop runs offline (provider: mock)
or as a cheap smoke test against a real API, before plugging in BIRD dev.
"""
import os
import sqlite3


def build_demo_db(path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE stadium(stadium_id INTEGER PRIMARY KEY, name TEXT, capacity INTEGER);
        CREATE TABLE singer(singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE concert(
            concert_id INTEGER PRIMARY KEY,
            stadium_id INTEGER,
            singer_id INTEGER,
            FOREIGN KEY(stadium_id) REFERENCES stadium(stadium_id),
            FOREIGN KEY(singer_id) REFERENCES singer(singer_id)
        );
        """
    )
    con.executemany("INSERT INTO stadium VALUES(?,?,?)",
                    [(1, "Bird Nest", 80000), (2, "Small Hall", 3000), (3, "Grand Arena", 55000)])
    con.executemany("INSERT INTO singer VALUES(?,?,?)",
                    [(1, "Alice", "US"), (2, "Bob", "UK"), (3, "Carol", "US")])
    con.executemany("INSERT INTO concert VALUES(?,?,?)",
                    [(1, 1, 1), (2, 3, 2), (3, 2, 3), (4, 1, 2)])
    con.commit()
    con.close()
    return path
