import os
from typing import Optional, List
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langgraph.prebuilt import create_react_agent

class SQLAnalyzer:
    def __init__(self, openai_api_key: str):
        """Initialize SQL Analyzer with necessary configurations."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0,
            presence_penalty=0.6,
            frequency_penalty=0.6
        )
        self.setup_database_connection()
        self.memory = ConversationBufferMemory(
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
    def setup_database_connection(self):
        """Set up the initial database connection."""
        username = os.getenv("DB_USER", "default_user")
        password = quote_plus(os.getenv("DB_PASSWORD", "default_password"))
        host = os.getenv("DB_HOST", "localhost")
        self.base_connection_string = f"mysql+pymysql://{username}:{password}@{host}:3306"
        
        try:
            self.engine = create_engine(
                self.base_connection_string,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            self.available_databases = self.get_available_databases()
            print("Database connection established successfully")
        except Exception as e:
            print(f"Error setting up database connection: {str(e)}")
            raise

    def get_available_databases(self) -> List[str]:
        """Get list of available databases excluding system databases."""
        system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys'}
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SHOW DATABASES"))
                return [row[0] for row in result if row[0] not in system_dbs]
        except Exception as e:
            print(f"Error fetching databases: {str(e)}")
            return []

    def connect_to_database(self, database_name: str) -> bool:
        """Connect to a specific database."""
        if database_name not in self.available_databases:
            print(f"Database '{database_name}' not found")
            return False

        try:
            # Create new connection string with specific database
            connection_string = f"{self.base_connection_string}/{database_name}"
            specific_engine = create_engine(connection_string)
            
            # Test connection
            with specific_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Initialize SQLDatabase instance and toolkit
            self.db = SQLDatabase(specific_engine)
            self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            
            # Set up the agent executor with latest prompt template
            prompt = hub.pull("langchain-ai/sql-agent-system-prompt")
            system_message = prompt.format(dialect="MySQL", top_k=5)
            
            self.agent_executor = create_react_agent(
                self.llm,
                self.toolkit.get_tools(),
                state_modifier=system_message
            )
            
            print(f"Successfully connected to {database_name}")
            return True
            
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            return False

    async def process_query(self, query: str) -> Optional[str]:
        """Process a natural language query and return the result."""
        if not query:
            return None
            
        try:
            # Get conversation context
            memory_vars = self.memory.load_memory_variables({"input": query})
            context = memory_vars.get("history", "")
            
            # Enhance query with context
            enhanced_query = f"""
            Previous context: {context}
            Current query: {query}
            """
            
            # Execute query through agent
            events = self.agent_executor.stream(
                {"messages": [("user", enhanced_query)]},
                stream_mode="values"
            )
            
            # Process response
            response = None
            for event in events:
                message = event["messages"][-1]
                if (
                    hasattr(message, 'content')
                    and not message.content.startswith('Tool Calls:')
                    and not message.content.startswith('Name: ')
                ):
                    response = message.content
            
            # Save to memory
            if response:
                self.memory.save_context(
                    {"input": query},
                    {"output": response}
                )
            
            return response
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return "Error processing your query. Please try again."

async def main():
        """Main function to run the SQL Analyzer."""
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            print("Missing OpenAI API key in environment variables")
            return

        try:
            # Initialize analyzer
            analyzer = SQLAnalyzer(openai_api_key)
            
            # Show available databases
            print("\nAvailable databases:", ", ".join(analyzer.available_databases))
            
            # Get database selection
            while True:
                db_name = input("\nEnter database name to connect (or 'exit' to quit): ")
                if db_name.lower() == 'exit':
                    break
                    
                if analyzer.connect_to_database(db_name):
                    # Main query loop
                    while True:
                        query = input("\nEnter your query (or 'back' to change database, 'exit' to quit): ")
                        
                        if query.lower() == 'exit':
                            return
                        if query.lower() == 'back':
                            break
                            
                        response = await analyzer.process_query(query)
                        if response:
                            print("\nResponse:", response)
                        
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        except Exception as e:
            print(f"\nFatal error: {str(e)}")
        finally:
            print("\nGoodbye!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())