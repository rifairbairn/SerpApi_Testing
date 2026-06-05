
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DatabaseHandler:
    def __init__(self, database: str):
        server = "DC1SQL06C"
        self.database_url = f"mssql+pyodbc://{server}/{database}?driver=SQL+Server&trusted_connection=yes"
        self.engine = create_engine(self.database_url, fast_executemany=True)
        self.Session = sessionmaker(bind=self.engine)

    def execute_query(self, query: str, params: dict = None, fetch: bool = False):
        """Executes a query with optional parameters and returns a pandas DataFrame if fetch=True."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params or {})
                if fetch:
                    # Fetch all rows
                    rows = result.fetchall()
                    # Get column names from the cursor metadata
                    columns = result.keys()
                    # Convert to pandas DataFrame
                    return pd.DataFrame(rows, columns=columns)

                connection.commit()  # Explicit commit for modifications

        except Exception as e:
            logging.error(f"Database query failed: {e}")
            return None

    def get_entity(self, entity_id: str = '', entity_name: str = ''):
        """Retrieves entity information based on entity ID or name."""
        query = """
        EXEC [dbo].[ROTHKO_Entity_Get] @EntityID = :entity_id, @EntityName = :entity_name
        """
        params = {"entity_id": entity_id, "entity_name": entity_name}
        return self.execute_query(query, params, fetch=True)

    def get_index_positions(self, ticker: str, pos_date: str):
        """Fetches historical constituents for a given portfolio ticker and date range."""

        query = """
        select c.Sedol, c.Allocation, e.EntityName, e.EntitySearchNames, e.Country, e.Sector
        from Rothkofo.dbo.ROTHKO_vConstituents c
        left join RothkoNLP.dbo.ROTHKO_Entity e on c.Sedol=e.EntityID
        where c.Ticker = :ticker and c.PositionDate = :pos_date
        """
        
        params = {"ticker": ticker, "pos_date": pos_date}
        return self.execute_query(query, params, fetch=True) 
        