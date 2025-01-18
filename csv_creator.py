import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from typing import List, Dict, Tuple, Optional
from urllib.parse import quote_plus
from dotenv import load_dotenv
from openai import OpenAI
import asyncio
import time

class CSVCreator:
    def __init__(self, db_config: Dict[str, str], openai_api_key: str):
        self.db_config = db_config
        self.client = OpenAI(api_key=openai_api_key)
        self.engine = None

    def connect_to_database(self, database_name: str) -> bool:
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                connection_string = (
                    f"mysql+pymysql://{self.db_config['user']}:{quote_plus(self.db_config['password'])}"
                    f"@{self.db_config['host']}:{self.db_config['port']}/{database_name}"
                )
                self.engine = create_engine(connection_string)
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                print(f"Connected to {database_name}")
                return True
            except Exception as e:
                print(f"Connection error (attempt {attempt + 1}): {str(e)}")
                time.sleep(2)  # Wait before retrying
        return False

    def get_database_schema(self) -> Dict:
        schema = {}
        try:
            with self.engine.connect() as conn:
                tables = conn.execute(text("SHOW TABLES"))
                for table in tables:
                    table_name = table[0]
                    columns = conn.execute(text(f"SHOW COLUMNS FROM {table_name}"))
                    schema[table_name] = [
                        {"name": col[0], "type": col[1], "key": col[3]} 
                        for col in columns
                    ]
            return schema
        except Exception as e:
            print(f"Error getting schema: {str(e)}")
            return {}

    async def analyze_query(self, query: str) -> Dict:
        try:
            schema = self.get_database_schema()
            
            prompt = f"""
            Given this user query: \"{query}\"
            And this database schema: {json.dumps(schema, indent=2)}

            Return ONLY a JSON object with:
            1. List of tables needed
            2. List of specific columns needed from each table
            3. Join conditions between tables (if multiple tables)

            Format:
            {{
                "tables": ["table1", "table2"],
                "columns": {{"table1": ["col1", "col2"], "table2": ["col1", "col2"]}},
                "joins": [{{"table1": "table1", "column1": "id", "table2": "table2", "column2": "id"}}]
            }}
            """

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a SQL expert. Respond only with valid JSON."},
                         {"role": "user", "content": prompt}],
                temperature=0
            )

            return json.loads(response.choices[0].message.content.strip())
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return None

    def generate_query(self, analysis: Dict) -> str:
        try:
            select_parts = []
            for table, columns in analysis["columns"].items():
                for col in columns:
                    select_parts.append(f"{table}.{col}")

            query_parts = [f"SELECT {', '.join(select_parts)}"]
            query_parts.append(f"FROM {analysis['tables'][0]}")

            if len(analysis['tables']) > 1 and analysis.get('joins'):
                for join in analysis['joins']:
                    query_parts.append(
                        f"JOIN {join['table2']} ON "
                        f"{join['table1']}.{join['column1']} = {join['table2']}.{join['column2']}"
                    )

            return "\n".join(query_parts)
            
        except Exception as e:
            print(f"Query generation error: {str(e)}")
            return None

    def check_and_remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Running duplication checking...")
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            print(f"Duplicate columns found and will be removed: {duplicate_columns}")
            df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
        else:
            print("No duplicate columns found.")
        return df

    async def create_csv(self, query: str, output_file: str = None) -> bool:
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                print("Analyzing query...")
                analysis = await self.analyze_query(query)

                if not analysis:
                    return False

                print("Generating SQL...")
                sql_query = self.generate_query(analysis)

                if not sql_query:
                    return False

                print("Executing query and creating CSV...")
                with self.engine.connect() as conn:
                    df = pd.read_sql(sql_query, conn)

                df = self.check_and_remove_duplicate_columns(df)

                if output_file is None:
                    output_file = f"output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

                df.to_csv(output_file, index=False)
                print(f"CSV created: {output_file}")
                print(f"Shape: {df.shape}")
                return True

            except Exception as e:
                print(f"CSV creation error (attempt {attempt + 1}): {str(e)}")
                time.sleep(2)  # Wait before retrying
        return False

async def main():
    load_dotenv()

    db_config = {
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '3306')
    }

    creator = CSVCreator(
        db_config=db_config,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

    database_name = input("Enter database name: ")
    if not creator.connect_to_database(database_name):
        print("Failed to connect to the database.")
        return

    while True:
        try:
            query = input("\nEnter your query (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break

            filename = input("Enter output filename (or press Enter for default): ")
            if not filename:
                filename = f"output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

            success = await creator.create_csv(query, filename)
            if not success:
                print("Failed to create CSV")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())