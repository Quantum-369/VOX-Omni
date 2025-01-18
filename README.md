# VOX Omni Analysis Expert: Advanced Multi-Agent AI Platform

VOX Omni Analysis Expert is a sophisticated multi-agent AI platform designed to streamline data analysis, competitor research, and intelligent query resolution. The platform integrates state-of-the-art models like Gemini 2.0 Flash, OpenAI's GPT models, and Claude for predictive, prescriptive, diagnostic, and descriptive analytics, enabling users to make data-driven decisions efficiently.

---

## Key Features

### 1. **Multi-Agent Architecture**
- **General Queries**: Handled by Gemini 2.0 Flash with its extensive token limit.
- **Data Analysis**:
  - **Descriptive Analysis**: Uses LangChain SQL toolkit to generate and execute SQL queries based on natural language inputs.
  - **Advanced Analytics** (Diagnostic, Predictive, Prescriptive): Integrates Claude to generate Python-based analytical solutions executed dynamically.
- **Competitor Analysis**: Employs Perplexity AI for web searches and generates actionable insights.

### 2. **Database Connectivity**
- Seamless integration with multiple database systems.
- Efficient query execution using SQLAnalyzer.

### 3. **CSV Generation**
- Dynamic generation of CSV files from database queries.
- Deduplication and validation ensure high-quality outputs.

### 4. **Enhanced Context Management**
- Maintains a robust history of interactions for contextual and accurate responses.

### 5. **Summarization and Insights**
- Summarization of outputs by Gemini 2.0 for clear, concise, and actionable results.

---

## System Workflow

### 1. **General Query Processing**
- Gemini 2.0 processes and directly responds to general queries.

### 2. **Data Query Handling**
#### Step 1: Intent Identification
- Queries are classified as general or data-related.

#### Step 2: Database Selection
- Users select from a list of connected databases.

#### Step 3: Analysis Type Determination
- Descriptive: Direct SQL execution using LangChain.
- Diagnostic, Predictive, Prescriptive: CSV creation followed by Python analysis with Claude.

#### Step 4: Summarization
- Results are sent to Gemini 2.0 for a summarized response.

### 3. **Competitor Analysis**
- Real-time insights generated via Perplexity AI with Gemini 2.0 summarization.

---

## Installation

### Prerequisites
- Python 3.9+
- Supported database (MySQL, PostgreSQL, etc.)
- API keys for:
  - OpenAI (GPT-4)
  - Anthropic (Claude)
  - Perplexity AI
  - Gemini 2.0

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/vectron.git
   ```
2. Navigate to the project directory:
   ```bash
   cd vectron
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   - Create a `.env` file:
     ```env
     OPENAI_API_KEY=<your-openai-api-key>
     ANTHROPIC_API_KEY=<your-anthropic-api-key>
     GEMINI_API_KEY=<your-gemini-api-key>
     PERPLEXITY_API_KEY=<your-perplexity-api-key>
     DB_USER=<your-database-user>
     DB_PASSWORD=<your-database-password>
     DB_HOST=<your-database-host>
     DB_PORT=<your-database-port>
     ```

---

## Usage

### Starting the Assistant
Run the main assistant:
```bash
python gemini_database_assistant.py
```

### Workflow Examples
#### General Query:
```
> What is the population of New York?
```
Response processed by Gemini 2.0.

#### Data Query:
```
> Show me the sales trends for the last quarter.
```
1. Database selection.
2. SQL execution or CSV generation with advanced analytics.

#### Competitor Research:
```
> What are the top features of my competitor's product?
```
Web search and summarization via Perplexity AI and Gemini.

---

## Project Structure
- **`csv_creator.py`**: Handles dynamic CSV generation from databases.
- **`sql_assistant.py`**: Processes SQL queries using LangChain and GPT.
- **`Successful_Analyser.py`**: Executes diagnostic, predictive, and prescriptive analytics.
- **`Integrated_analyser.py`**: Combines CSV creation and analysis workflows.
- **`function_library.py`**: Utility functions for token usage, query handling, and more.
- **`gemini_database_assistant.py`**: Entry point for the assistant.
- **`perplexity_agent.py`**: Integrates Perplexity AI for competitor insights.

---

## Contributing
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact
For inquiries or support, please reach out to:
- **Name**: Harsha Vardhan Sai Machineni

