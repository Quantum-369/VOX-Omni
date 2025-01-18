# function_library.py
from perplexity_agent import PerplexityAI

async def handle_perplexity_query(question: str, model: str = "llama-3.1-sonar-small-128k-online"):
    """Send a query to Perplexity AI and return the response."""
    perplexity_client = PerplexityAI()  # Automatically fetches API key from .env
    result = perplexity_client.query(question=question, model=model)
    
    if result:
        # Extract the response content
        completion = result.get("choices", [{}])[0].get("message", {}).get("content", "No response found.")
        return completion
    else:
        return "Unable to fetch data. Please try again."
async def handle_csv_creation(csv_creator, database_name, query, output_file):
    """Handle CSV creation and return the status."""
    if not csv_creator.connect_to_database(database_name):
        return f"Failed to connect to database: {database_name}"
    
    success = await csv_creator.create_csv(query, output_file)
    return "CSV created successfully." if success else "Failed to create CSV."

async def handle_query(sql_analyzer, database_name, query):
    """Handle SQL queries and return raw results."""
    if not sql_analyzer.connect_to_database(database_name):
        return f"Failed to connect to database: {database_name}"
    
    raw_response = await sql_analyzer.process_query(query)
    if not raw_response:
        return "No response received."
    
    return raw_response

async def handle_advanced_analysis(database_name, question, integrated_analyzer):
    """Handle advanced analysis using Integrated Analyzer."""
    # Connect to the database
    if not integrated_analyzer.connect_to_database(database_name):
        return f"Failed to connect to database: {database_name}"
    
    # Perform the analysis, including any necessary CSV creation
    result = await integrated_analyzer.analyze(question)
    if not result.get('success', False):
        return f"Analysis failed: {result.get('error', 'Unknown error')}"
    
    return result.get('summary', 'Analysis completed successfully.')
def display_token_usage(chat, response):
    """Display token usage details."""
    usage_metadata = response.usage_metadata
    prompt_token_count = usage_metadata.prompt_token_count
    candidates_token_count = usage_metadata.candidates_token_count
    total_token_count = chat.model.count_tokens(chat.history)

    token_details = (
        f"\nPrompt Tokens Used: {prompt_token_count}\n"
        f"Response Tokens Used: {candidates_token_count}\n"
        f"Total Tokens Used: {total_token_count}\n"
    )
    print(token_details)
    return token_details

# Add function declarations for tools
tool_function_declarations = [
    {
        "name": "run_sql_query",
        "description": "Run a natural language SQL query on a specified database and retrieve results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "database_name": {"type": "string"}
            },
            "required": ["query", "database_name"]
        }
    },
    {
        "name": "query_perplexity",
        "description": "Query Perplexity AI for real-time weather, web search, or general information.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The user's question for Perplexity AI."},
                "model": {"type": "string", "description": "The model to use (optional)."}
            },
            "required": ["question"]
        }
    },
    {
        "name": "display_token_usage",
        "description": "Display token usage details, including prompt, response, and total tokens.",
        "parameters": {
            "type": "object",
            "properties": {
                "dummy": {
                    "type": "string",
                    "description": "A dummy parameter for validation purposes."
                }
            },
            "required": ["dummy"]
        }
    },
    {
        "name": "run_advanced_analysis",
        "description": "Perform advanced analysis on a specified database using a natural language question.",
        "parameters": {
            "type": "object",
            "properties": {
                "database_name": {"type": "string"},
                "question": {"type": "string"}
            },
            "required": ["database_name", "question"]
        }
    },
    {
        "name": "create_csv",
        "description": "Create a CSV file based on a natural language query and specified database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "database_name": {"type": "string"},
                "output_file": {"type": "string"}
            },
            "required": ["query", "database_name"]
        }
    }
]
