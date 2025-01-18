import os
import asyncio
import google.generativeai as genai
from sql_assistant import SQLAnalyzer
from csv_creator import CSVCreator
from function_library import handle_csv_creation, handle_query, display_token_usage, tool_function_declarations, handle_advanced_analysis
from Integrated_analyser import EnhancedDataAnalyzer
from function_library import handle_perplexity_query
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = genai.GenerationConfig(
    temperature=0,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    tools=[{
        "function_declarations": tool_function_declarations
    }]
)

async def main():
    print("Welcome to the Gemini Assistant!")
    print("Initializing analyzers...")

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

    print("\nAvailable databases:", ", ".join(sql_analyzer.available_databases))

    chat = model.start_chat(history=[])

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "q":
            print("Exiting the assistant. Goodbye!")
            break

        try:
            response = chat.send_message(user_input)
            display_token_usage(chat, response)

            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.function_call:
                        fn = part.function_call
                        if fn.name == "run_sql_query":
                            db_name = fn.args.get("database_name", "")
                            query = fn.args.get("query", "")
                            sql_response = await handle_query(sql_analyzer, db_name, query)
                            gemini_summary = chat.send_message(
                                f"Summarize the following SQL query result:\n{sql_response}"
                            )
                            summarized_response = gemini_summary.candidates[0].content.parts[0].text
                            print("\nSummarized SQL Query Response:", summarized_response)
                        elif fn.name == "create_csv":
                            db_name = fn.args.get("database_name", "")
                            query = fn.args.get("query", "")
                            output_file = fn.args.get("output_file", "output.csv")
                            csv_response = await handle_csv_creation(csv_creator, db_name, query, output_file)
                            print("\nCSV Creation Response:", csv_response)
                        elif fn.name == "query_perplexity":
                            question = fn.args.get("question", "")
                            perplexity_model = fn.args.get("model", "llama-3.1-sonar-small-128k-online")  # Rename this variable
                            perplexity_response = await handle_perplexity_query(question=question, model=perplexity_model)
                            #print("\nSummarized Perplexity Response:", perplexity_response)
                            # Summarize the Perplexity response using Gemini
                            gemini_summary = chat.send_message(
                                f"Summarize the following information:\n{perplexity_response}"
                            )
                            summarized_response = gemini_summary.candidates[0].content.parts[0].text
                            print("\nFrom Perplexity:", summarized_response)
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
                                summarized_response = gemini_summary.candidates[0].content.parts[0].text
                                print("\nGemini Response:", summarized_response)
                            else:
                                print("\nCSV creation failed. Advanced analysis cannot proceed.")
                    else:
                        print("\nGemini:", part.text)

        except AttributeError as e:
            print("Attribute Error:", e)
        except Exception as e:
            print(f"Error processing request: {e}")

if __name__ == "__main__":
    asyncio.run(main())
