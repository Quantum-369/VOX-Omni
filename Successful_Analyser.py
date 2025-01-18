import os
import subprocess
import json
import sys
import tempfile
import pandas as pd
import requests
from typing import Dict, Any, Tuple, Optional, List
from dotenv import load_dotenv
import asyncio
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMClient:
    """Handles communication with Claude API"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
        self.headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
        self.base_url = 'https://api.anthropic.com/v1/messages'
        
    async def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response from Claude with retry logic"""
        data = {
            'model': 'claude-3-haiku-20240307',
            'max_tokens': 1024,
            'system': 'You are a helpful data analysis assistant that writes Python code.',
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    text = response.json()['content'][0]['text']
                    
                    # Clean up the response
                    if "import" in text:
                        text = text[text.find("import"):]
                    text = text.replace('```python', '').replace('```', '')
                    if "import" in text:
                        lines = text.split('\n')
                        code_lines = []
                        for line in lines:
                            if line.strip() and not line.startswith(('Here', 'This', 'Note')):
                                code_lines.append(line)
                        text = '\n'.join(code_lines)
                    return text.strip()
                    
                elif response.status_code == 429:
                    wait_time = min(2 ** attempt, 8)
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    response.raise_for_status()
                    
            except Exception as e:
                logger.error(f"Error calling Claude API: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)
                
        raise Exception("Failed to get response from Claude after all retries")

class CodeExecutor:
    """Handles safe execution of generated code"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        
    def execute(self, code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute code in isolation and return results"""
        
        # Create temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create code file
            code_path = os.path.join(temp_dir, 'analysis.py')
            with open(code_path, 'w') as f:
                f.write(code)
            
            try:
                # Execute in subprocess with timeout
                result = subprocess.run(
                    [sys.executable, code_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=temp_dir  # Run in temp directory
                )
                
                if result.returncode == 0:
                    return True, result.stdout, None
                else:
                    return False, None, result.stderr
                    
            except subprocess.TimeoutExpired:
                return False, None, f"Code execution timed out after {self.timeout} seconds"
            except Exception as e:
                return False, None, str(e)

class DataAnalyzer:
    """Main class for data analysis"""
    
    def __init__(self, csv_path: str, max_retries: int = 3):
        self.csv_path = csv_path.replace('\\', '\\\\')
        self.max_retries = max_retries
        self.llm_client = LLMClient()
        self.executor = CodeExecutor()
        
        # Initialize data info
        self._initialize_data_info()
        
    def _initialize_data_info(self):
        """Get dataset information"""
        try:
            #df = pd.read_csv(r'C:\Users\harsh\Downloads\MS projects\descriptive_analytics_engine\working_test_files\Dataset.csv', encoding='utf-8')
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            self.data_info = {
                'path': self.csv_path,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample': df.to_dict(orient='records'),
                'total_rows': sum(1 for _ in open(self.csv_path, encoding='utf-8')) - 1 # Excluding header
            }
            
            # Get basic stats for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                stats_df = pd.read_csv(self.csv_path, usecols=numeric_cols)
                self.data_info['numeric_stats'] = {
                    col: {
                        'min': stats_df[col].min(),
                        'max': stats_df[col].max(),
                        'mean': stats_df[col].mean()
                    } for col in numeric_cols
                }
                
        except Exception as e:
            logger.error(f"Error initializing data info: {str(e)}")
            raise
    def _get_descriptive_prompt(self, question: str, error: Optional[str] = None) -> str:
        escaped_path = self.csv_path.replace('\\', '\\\\')
        limited_sample = self.data_info['sample'][:2]
        
        return f"""Generate Python code for descriptive analysis: {question}
        
import pandas as pd
import numpy as np
import json

Analyze the CSV at: r'{escaped_path}'
Columns: {', '.join(self.data_info['columns'])}
Types: {json.dumps(self.data_info['dtypes'])}
Sample: {json.dumps(limited_sample)}

Previous error: {error}

Include:
- Data loading and validation
- Summary statistics
- Clear JSON output format
- convert data types into json serializable format
- Error handling
Remember: Start with 'import' and include NO explanatory text, NO markdown, NO additional content."""

    def _get_diagnostic_prompt(self, question: str, error: Optional[str] = None) -> str:
        escaped_path = self.csv_path.replace('\\', '\\\\')
        
        return f"""Generate Python code for diagnostic analysis: {question}
        
import pandas as pd
import numpy as np
from scipy import stats
import json

Analyze the CSV at: r'{escaped_path}'
Columns: {', '.join(self.data_info['columns'])}

Include:
- Statistical tests
- Correlation analysis
- Root cause identification
- Clear JSON output
- convert data types into json serializable format
- Error handling

Previous error: {error}
Remember: Start with 'import' and include NO explanatory text, NO markdown, NO additional content."""

    def _get_predictive_prompt(self, question: str, error: Optional[str] = None) -> str:
        escaped_path = self.csv_path.replace('\\', '\\\\')
        
        return f"""Generate Python code to address this predictive analysis question: {question}

    Analyze the CSV at: r'{escaped_path}'
    Available columns in dataset: {', '.join(self.data_info['columns'])}
    Data types: {json.dumps(self.data_info['dtypes'])}

    Required imports:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor  # For complex relationships
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    import json
    from datetime import datetime

    Key requirements:
    1. Identify target variable from the question
    2. Select relevant features for prediction
    3. Handle both categorical and numerical data appropriately
    4. Process date fields if relevant
    5. Scale/normalize features as needed
    6. Choose appropriate model based on prediction type
    7. Provide model performance metrics
    8. Return results in JSON format with error handling

    Example output format:
    {{
        'result': {{
            'model_performance': {{
                'r2_score': float,
                'mse': float,
                'accuracy': float  # if applicable
            }},
            'predictions': list,
            'feature_importance': list,
            'model_details': str
        }}
    }}

    Previous error: {error}

    Important: 
    - Analyze the question to determine target variable
    - Select features based on their relevance to the prediction task
    - Choose appropriate model type (regression/classification)
    - Handle data types appropriately
    - convert data types into json serializable format
    - Include proper error handling
    Remember: Start with 'import' and include NO explanatory text, NO markdown, NO additional content."""

    def _get_prescriptive_prompt(self, question: str, error: Optional[str] = None) -> str:
        escaped_path = self.csv_path.replace('\\', '\\\\')
        
        return f"""Generate Python code for prescriptive analysis: {question}
        
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import json

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

Analyze the CSV at: r'{escaped_path}'
Columns: {', '.join(self.data_info['columns'])}

Required elements:
1. Load and validate data
2. Define optimization objective function
3. Set up constraints
4. Solve optimization problem
5. Format results with convert_to_serializable
- convert data types into json serializable format
6. Return JSON with:
   - Summary metrics
   - Recommendations
   - Action items
   
7. Handle all errors and edge cases

Previous error: {error}
Remember: Start with 'import' and include NO explanatory text, NO markdown, NO additional content."""

    def _detect_analysis_type(self, question: str) -> str:
        """Detect the type of analysis based on question keywords"""
        question = question.lower()
        
        predictive_keywords = ['predict', 'forecast', 'future', 'will', 'expect', 'trend']
        diagnostic_keywords = ['why', 'cause', 'reason', 'correlation', 'relationship', 'compare', 'impact']
        prescriptive_keywords = ['should', 'recommend', 'optimize', 'best course', 'action', 'improve', 'strategy']
        
        if any(keyword in question for keyword in predictive_keywords):
            return "predictive"
        elif any(keyword in question for keyword in diagnostic_keywords):
            return "diagnostic"
        elif any(keyword in question for keyword in prescriptive_keywords):
            return "prescriptive"
        else:
            return "descriptive"

    def _generate_code_prompt(self, question: str, error: Optional[str] = None) -> str:
        """Generate appropriate code based on the type of analysis needed"""
        analysis_type = self._detect_analysis_type(question)
        
        if analysis_type == "diagnostic":
            return self._get_diagnostic_prompt(question, error)
        elif analysis_type == "predictive":
            return self._get_predictive_prompt(question, error)
        elif analysis_type == "prescriptive":
            return self._get_prescriptive_prompt(question, error)
        else:
            return self._get_descriptive_prompt(question, error)        
    def _generate_summary_prompt(self, question: str, results: str) -> str:
        """Generate prompt for summarizing results based on analysis type and specific question"""
        analysis_type = self._detect_analysis_type(question)
        
        base_prompt = f"""Analyze these results specifically to answer: {question}

    Results to analyze:
    {results}

    Important guidelines:
    1. Only use data explicitly shown in the results above
    2. Focus on answering the specific question asked
    3. Do not include information not present in the results
    4. Use exact numbers and metrics from the results"""

        if analysis_type == "descriptive":
            base_prompt += f"""
    Based only on the numerical results provided, answer "{question}" by including:
    - Exact statistics and metrics from the results that address the question
    - Only patterns directly observable in the data
    - Specific data points relevant to the question
    Limit to 20 words. Include only information shown in the results that directly answers the question."""

        elif analysis_type == "diagnostic":
            base_prompt += f"""
    Based only on the numerical results provided, answer "{question}" by including:
    - Specific correlations from the results that address the question
    - Statistical significance values relevant to the question
    - Measured relationships that explain the asked phenomenon
    Limit to 20 words. Include only information shown in the results that directly answers the question."""

        elif analysis_type == "predictive":
            base_prompt += f"""
    Based only on the numerical results provided, answer "{question}" by including:
    - Model performance metrics (R², MSE) relevant to the prediction asked
    - Feature importance values that relate to the question
    - Specific prediction values that answer the question
    Limit to 20 words. Include only information shown in the results that directly answers the question."""

        elif analysis_type == "prescriptive":
            base_prompt += f"""
    Based only on the numerical results provided, answer "{question}" by including:
    - Quantified impacts relevant to the asked recommendation
    - Optimization results that address the question
    - Specific numerical recommendations that answer the question
    Limit to 20 words. Include only information shown in the results that directly answers the question."""

        base_prompt += """
    If any requested element is not present in the results, omit it rather than making assumptions.
    Focus solely on answering the specific question using available data."""

        return base_prompt

    async def analyze(self, question: str) -> Dict[str, Any]:
        """Main method to analyze data based on question"""
        
        attempt = 0
        last_error = None
        start_time = datetime.now()
        
        while attempt < self.max_retries:
            try:
                # Generate code
                code_prompt = self._generate_code_prompt(question, last_error)
                code = await self.llm_client.generate(code_prompt)
                
                # Execute code
                success, output, error = self.executor.execute(code)
                
                if success:
                     # Print generated code and execution results
                    print("\nGenerated Code:")
                    print("-" * 50)
                    print(code)
                    print("\nExecution Results:")
                    print("-" * 50)
                    print(output)
                    # Generate summary
                    summary_prompt = self._generate_summary_prompt(question, output)
                    summary = await self.llm_client.generate(summary_prompt)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        'success': True,
                        'summary': summary,
                        'raw_output': output,
                        'code': code,
                        'execution_time': execution_time,
                        'attempts': attempt + 1
                    }
                else:
                    last_error = error
                    attempt += 1
                    logger.warning(f"Attempt {attempt} failed: {error}")
                    
            except Exception as e:
                last_error = str(e)
                attempt += 1
                logger.error(f"Error in attempt {attempt}: {str(e)}")
                
        execution_time = (datetime.now() - start_time).total_seconds()
        return {
            'success': False,
            'error': last_error,
            'execution_time': execution_time,
            'attempts': attempt
        }

async def main():
    # Example usage
    try:
        #analyzer = DataAnalyzer(r'C:\Users\harsh\Downloads\MS projects\descriptive_analytics_engine\working_test_files\Dataset.csv')
        # For standalone use, get CSV path from user input
        if len(sys.argv) == 1:  # No command line arguments
            csv_path = input("Enter the path to your CSV file: ").strip()
        else:  # CSV path provided as argument (for integrated use)
            csv_path = sys.argv[1]
            
        analyzer = DataAnalyzer(csv_path)
        while True:
            # Ask user for a question
            question = input("\nEnter your question about the dataset (or type 'exit' to quit): ").strip()
            
            # Check if user wants to exit
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nExiting the program. Goodbye!")
                break
                
            # Skip empty questions
            if not question:
                print("Please enter a valid question.")
                continue
                
            logger.info(f"\nAnalyzing question: {question}")
            result = await analyzer.analyze(question)
            
            if result['success']:
                print(f"\nQuestion: {question}")
                print("\nSummary:")
                print(result['summary'])
                print(f"\nExecution time: {result['execution_time']:.2f} seconds")
                print(f"Attempts: {result['attempts']}")
                # print("\nFull Analysis Details:")
                # print("-" * 50)
                # print(f"Generated Code:\n{result['code']}")
                # print(f"\nRaw Output:\n{result['raw_output']}")
            else:
                print(f"\nAnalysis failed: {result['error']}")
            
            # Ask if user wants to continue
            continue_analysis = input("\nWould you like to ask another question? (yes/no): ").strip().lower()
            if continue_analysis not in ['yes', 'y']:
                print("\nExiting the program. Goodbye!")
                break
                
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())