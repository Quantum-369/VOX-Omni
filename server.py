from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import asyncio
from gemini_database_assistant import model
from sql_assistant import SQLAnalyzer
from csv_creator import CSVCreator
from function_library import (
    handle_query,
    handle_csv_creation,
    handle_perplexity_query,
    handle_advanced_analysis,
    display_token_usage
)
from Integrated_analyser import EnhancedDataAnalyzer
import os
from asgiref.sync import async_to_sync
from functools import partial

app = Flask(__name__, static_folder='.')
CORS(app)

# Initialize components
chat = model.start_chat(history=[])
sql_analyzer = SQLAnalyzer(openai_api_key=os.environ["OPENAI_API_KEY"])
csv_creator = CSVCreator(
    db_config={
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '3306')
    },
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

async def process_function_call(fn):
    """Process a function call from Gemini"""
    try:
        if fn.name == "run_sql_query":
            db_name = fn.args.get("database_name", "")
            query = fn.args.get("query", "")
            sql_response = await handle_query(sql_analyzer, db_name, query)
            gemini_summary = chat.send_message(
                f"Summarize the following SQL query result:\n{sql_response}"
            )
            return gemini_summary.text

        elif fn.name == "create_csv":
            db_name = fn.args.get("database_name", "")
            query = fn.args.get("query", "")
            output_file = fn.args.get("output_file", "output.csv")
            csv_response = await handle_csv_creation(csv_creator, db_name, query, output_file)
            return f"CSV Creation Response: {csv_response}"

        elif fn.name == "query_perplexity":
            question = fn.args.get("question", "")
            perplexity_model = fn.args.get("model", "llama-3.1-sonar-small-128k-online")
            perplexity_response = await handle_perplexity_query(question=question, model=perplexity_model)
            gemini_summary = chat.send_message(
                f"Summarize the following information:\n{perplexity_response}"
            )
            return f"From Perplexity: {gemini_summary.text}"

        elif fn.name == "run_advanced_analysis":
            db_name = fn.args.get("database_name", "")
            question = fn.args.get("question", "")
            output_file = os.path.abspath("temp_analysis.csv")
            
            csv_response = await handle_csv_creation(csv_creator, db_name, question, output_file)
            if csv_response == "CSV created successfully.":
                integrated_analyzer = EnhancedDataAnalyzer(csv_path=output_file)
                analysis_response = await integrated_analyzer.analyze(question)
                gemini_summary = chat.send_message(
                    f"Summarize the following Advanced analysis result:\n{analysis_response}"
                )
                return f"Advanced Analysis: {gemini_summary.text}"
            return "CSV creation failed. Advanced analysis cannot proceed."

        return f"Unknown function: {fn.name}"
    except Exception as e:
        print(f"Error in function {fn.name}: {str(e)}")
        return f"Error processing {fn.name}: {str(e)}"

async def process_response(response):
    """Process the complete response from Gemini"""
    try:
        result = []
        
        # Display token usage
        token_details = display_token_usage(chat, response)
        result.append(token_details)

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.function_call:
                    # Process function call and get result
                    fn_result = await process_function_call(part.function_call)
                    result.append(fn_result)
                else:
                    result.append(part.text)
        
        return "\n".join(filter(None, result))
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        return f"Error processing response: {str(e)}"

@app.route('/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.json
        message = data.get('message')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Send message to Gemini
        response = chat.send_message(message)
        
        # Process response asynchronously
        result = async_to_sync(process_response)(response)
        
        return jsonify({'response': result})

    except Exception as e:
        print(f"Error in chat handler: {str(e)}")
        return jsonify({'error': str(e)}), 500

def run_async_app():
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':
    run_async_app()