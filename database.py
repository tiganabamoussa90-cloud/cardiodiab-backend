import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

load_dotenv()  # Charge les variables du fichier .env en local ; ignoré si absent (ex: sur Render)

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.environ.get("DB_HOST"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            database=os.environ.get("DB_NAME"),
            port=int(os.environ.get("DB_PORT", 3306)),
            charset="utf8mb4"
        )
        if connection.is_connected():
            print("Connexion MySQL réussie")
            return connection
    except Error as e:
        print(f"Erreur MySQL : {e}")
    return None