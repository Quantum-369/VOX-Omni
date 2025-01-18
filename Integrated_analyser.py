import os
import asyncio
import tempfile
import pandas as pd
from dotenv import load_dotenv
from csv_creator import CSVCreator
from Successful_Analyser import DataAnalyzer

class EnhancedDataAnalyzer(DataAnalyzer):
    def __init__(self, csv_path: str, max_retries: int = 3):
        super().__init__(csv_path, max_retries)

async def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize CSV Creator
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
        
        # Get database name and connect
        database_name = input("Enter database name: ")
        if not creator.connect_to_database(database_name):
            print("Failed to connect to database")
            return
            
        while True:
            # Get query for CSV creation and analysis
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            # Generate CSV filename with absolute path
            output_file = os.path.abspath(f"output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            # Create CSV
            print("\nAnalyzing and creating CSV...")
            success = await creator.create_csv(query, output_file)
            if not success:
                print("Failed to create CSV")
                continue
            
            print(f"\nCSV created successfully: {output_file}")
            
            try:
                # Initialize enhanced analyzer with created CSV
                analyzer = EnhancedDataAnalyzer(output_file)
                
                # Run analysis with the same query
                print("\nAnalyzing the data...")
                result = await analyzer.analyze(query)
                
                # Display results
                if result['success']:
                    print("\nAnalysis Results:")
                    print("-" * 50)
                    print(result['summary'])
                    print(f"\nExecution time: {result['execution_time']:.2f} seconds")
                else:
                    print(f"\nAnalysis failed: {result['error']}")
                
                # Additional analysis loop
                while True:
                    additional_question = input("\nWould you like to ask another question about this data? (yes/no): ").strip().lower()
                    if additional_question not in ['yes', 'y']:
                        break
                        
                    question = input("\nEnter your additional analysis question: ")
                    result = await analyzer.analyze(question)
                    
                    if result['success']:
                        print("\nAnalysis Results:")
                        print("-" * 50)
                        print(result['summary'])
                        print(f"\nExecution time: {result['execution_time']:.2f} seconds")
                    else:
                        print(f"\nAnalysis failed: {result['error']}")
                        
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
                continue
                    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
