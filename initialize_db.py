import os
from dotenv import load_dotenv
from db import init_db

# Load environment variables before anything else
load_dotenv()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database tables created successfully!")
