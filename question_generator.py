import sqlite3
import pandas as pd
import numpy as np
from litellm import completion
import os
import sys
from pathlib import Path
import random
import time

# Set up the API key
os.environ['GROQ_API_KEY'] = "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"

def check_database_exists():
    """Check if the database file exists"""
    if not Path('data.db').exists():
        print("Error: data.db not found. Please run the main app and upload a CSV file first.")
        sys.exit(1)

def safe_float_format(value):
    """Safely format float values"""
    try:
        return f"{float(value):.2f}"
    except:
        return str(value)

def generate_data_context(df, table_name):
    """Generate context about the database for the LLM"""
    try:
        context = f"Table Name: {table_name}\n\n"
        
        # Add basic info
        context += f"Total Records: {len(df)}\n"
        context += f"Total Columns: {len(df.columns)}\n\n"
        
        # Add column information
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            context += "Numeric Columns:\n"
            for col in numeric_cols:
                stats = df[col].describe()
                context += f"- {col} (min: {stats['min']:.2f}, max: {stats['max']:.2f})\n"
        
        if len(categorical_cols) > 0:
            context += "\nCategorical Columns:\n"
            for col in categorical_cols:
                unique_vals = df[col].nunique()
                context += f"- {col} ({unique_vals} unique values)\n"
        
        return context
    except Exception as e:
        print(f"Error in generate_data_context: {str(e)}")
        return None

def generate_questions_with_llm(data_context):
    """Generate questions using LLM"""
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a data analyst. Generate 5 analytical questions about the given database.
                For each question, include:
                1. The question itself
                2. Suggested visualization (choose from: bar, pie, scatter, line, histogram, box)
                3. Brief purpose
                
                Format exactly as:
                Q: [question]
                V: [visualization]
                P: [purpose]
                ---"""
            },
            {
                "role": "user",
                "content": f"Generate questions for this database:\n{data_context}"
            }
        ]
        
        response = completion(
            model="groq/llama3-8b-8192",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generate_questions_with_llm: {str(e)}")
        return None

def parse_questions(response_text):
    """Parse LLM response into structured questions"""
    if not response_text:
        return []
    
    questions = []
    current_question = {}
    
    try:
        parts = response_text.split('---')
        for part in parts:
            if not part.strip():
                continue
            
            lines = part.strip().split('\n')
            question_dict = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    question_dict['question'] = line[2:].strip()
                elif line.startswith('V:'):
                    question_dict['visualization'] = line[2:].strip()
                elif line.startswith('P:'):
                    question_dict['purpose'] = line[2:].strip()
            
            if question_dict.get('question'):
                questions.append(question_dict)
        
        return questions
    except Exception as e:
        print(f"Error parsing questions: {str(e)}")
        return []

def display_questions(questions):
    """Display the generated questions in a formatted way"""
    if not questions:
        print("No questions were generated.")
        return
        
    print("\n=== Generated Questions for Your Database ===\n")
    
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(f"Q: {q.get('question', 'No question provided')}")
        print(f"Suggested Visualization: {q.get('visualization', 'Not specified')}")
        print(f"Purpose: {q.get('purpose', 'Not specified')}")
        print("-" * 80)

def detect_join_relationships(conn):
    """Detect potential join relationships between tables by looking for common column names"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables = [t[0] for t in tables]
    
    relationships = []
    table_columns = {}
    
    # Get columns for each table
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        table_columns[table] = column_names
    
    # Find potential join keys between tables
    for i, table1 in enumerate(tables):
        for j, table2 in enumerate(tables):
            if i >= j:  # Skip same table and avoid duplicates
                continue
                
            common_columns = set(table_columns[table1]) & set(table_columns[table2])
            
            # Look for potential ID fields
            id_columns = [col for col in common_columns if 
                          col.lower().endswith('id') or 
                          col.lower() == 'id' or
                          col.lower() == table1.lower() + '_id' or
                          col.lower() == table2.lower() + '_id']
            
            if id_columns:
                for col in id_columns:
                    relationships.append({
                        'table1': table1,
                        'table2': table2,
                        'key': col,
                        'relationship': f"{table1}.{col} = {table2}.{col}"
                    })
            elif common_columns:
                # Use the first common column if no ID columns found
                col = list(common_columns)[0]
                relationships.append({
                    'table1': table1,
                    'table2': table2,
                    'key': col,
                    'relationship': f"{table1}.{col} = {table2}.{col}"
                })
    
    return relationships, table_columns

def get_suggested_questions(df, table_name, dml_enabled=False, join_relationships=None):
    """Generate suggested questions based on the data and DML mode, including JOIN questions"""
    
    # Analyze column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Analysis focus options
    analysis_focuses = [
        "trends and patterns",
        "comparative analysis",
        "statistical insights",
        "performance metrics",
        "distribution analysis",
        "relationship studies",
        "outlier detection",
        "category breakdown",
        "temporal analysis",
        "correlation studies",
        "cross-table analysis"  # New focus for JOIN operations
    ]
    
    # Randomly select focus areas and ensure they're different each time
    current_focus = random.sample(analysis_focuses, 3)
    
    # Generate unique timestamp for this request
    timestamp = int(time.time() * 1000)  # Using milliseconds for more uniqueness
    
    # Create dynamic context with random elements
    data_context = f"""
    Table Name: {table_name}
    Request ID: {timestamp}
    Analysis Focus Areas: {', '.join(current_focus)}
    
    Available Columns for Analysis:
    - Numeric Columns: {', '.join(random.sample(numeric_cols, len(numeric_cols)))}
    - Categorical Columns: {', '.join(random.sample(categorical_cols, len(categorical_cols)))}
    - Date Columns: {', '.join(date_cols) if date_cols else 'None'}
    
    Data Statistics:
    {df.describe().to_string() if len(numeric_cols) > 0 else 'No numeric columns'}
    """
    
    # Add join relationship context if available
    if join_relationships:
        join_context = "Available Join Relationships:\n"
        for rel in join_relationships:
            join_context += f"- {rel['table1']} can be joined with {rel['table2']} on {rel['key']}\n"
        data_context += f"\n{join_context}"
    
    if dml_enabled:
        system_prompt = f"""You are a data analyst generating unique questions for request {timestamp}.
        Current analysis focuses: {', '.join(current_focus)}
        
        Generate completely new questions that include:
        1. Data analysis questions with visualizations
        2. Data modification questions
        3. JOIN operation questions across tables (if relationships exist)
        
        Each question must be:
        - Unique and specific to this request
        - Different from any previous questions
        - Relevant to the current focus areas
        - Using actual column names from the data
        
        For each question, specify:
        1. A clear, unique question
        2. Visualization type (for analytical questions)
        3. A concise purpose (6-7 lines)
        4. Operation type (read/modify/join)"""
        
        user_prompt = f"""Using this data context:
        {data_context}
        
        Generate 8 unique questions:
        - 3 analytical questions with different visualization types
        - 3 different types of data modification questions
        - 2 JOIN operations across tables (if join relationships exist)
        
        Return as a list of dictionaries:
        {{
            "question": "specific, unique question",
            "visualization": "visualization type",
            "purpose": "6-7 lines of purpose",
            "type": "read or modify or join"
        }}
        
        Ensure each question is different and uses specific column names."""
        
    else:
        system_prompt = f"""You are a data analyst generating unique questions for request {timestamp}.
        Current analysis focuses: {', '.join(current_focus)}
        
        Generate completely new analytical questions that:
        - Are unique to this specific request
        - Use different visualization types
        - Focus on different aspects of the data
        - Use actual column names
        - Include JOIN operations where applicable
        
        Each question must specify:
        1. A clear, unique question
        2. Appropriate visualization type
        3. Concise purpose (6-7 lines)
        4. Type (read or join)"""
        
        user_prompt = f"""Using this data context:
        {data_context}
        
        Generate 7 unique analytical questions:
        - 5 standard analytical questions with different visualizations
        - 2 JOIN operations questions to analyze across tables (if relationships exist)
        
        Return as a list of dictionaries:
        {{
            "question": "specific, unique question",
            "visualization": "visualization type",
            "purpose": "6-7 lines of purpose",
            "type": "read or join"
        }}"""

    try:
        # Use high temperature and unique seed for more variation
        response = completion(
            model="groq/llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.9,  # High temperature for more randomness
            seed=timestamp  # Unique seed for each request
        )

        questions = eval(response.choices[0].message.content)
        
        # Add metadata to track uniqueness
        for q in questions:
            q['generated_at'] = timestamp
            q['focus_areas'] = current_focus

        return questions

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        # Fallback questions with randomization including JOIN if relationships exist
        fallback_questions = [
            {
                "question": f"Show the distribution of {random.choice(numeric_cols)}" if numeric_cols else "Show all records",
                "visualization": "histogram" if numeric_cols else "table",
                "purpose": f"""Analyzing the distribution pattern to identify key trends and outliers.
Understanding data concentration and spread across different values.
Identifying potential anomalies or unusual patterns in the data.
Supporting data-driven decision making with statistical insights.
Helping establish normal ranges and benchmark values.
Guiding strategy development and goal setting.
Enabling better understanding of data characteristics.""",
                "type": "read",
                "generated_at": timestamp,
                "focus_areas": current_focus
            },
            {
                "question": f"Compare {random.choice(numeric_cols)} across {random.choice(categorical_cols)}" if (numeric_cols and categorical_cols) else "Show record count",
                "visualization": "bar",
                "purpose": f"""Identifying performance variations across different categories.
Highlighting top and bottom performing segments clearly.
Revealing opportunities for targeted improvements.
Supporting strategic decision making and resource allocation.
Providing insights for category-specific strategies.
Understanding relative performance metrics.
Guiding optimization efforts across categories.""",
                "type": "read",
                "generated_at": timestamp,
                "focus_areas": current_focus
            }
        ]
        
        # Add JOIN fallback question if relationships exist
        if join_relationships:
            join_rel = random.choice(join_relationships)
            fallback_questions.append({
                "question": f"Join {join_rel['table1']} and {join_rel['table2']} on {join_rel['key']} to analyze related data",
                "visualization": "table",
                "purpose": f"""Connecting related data across tables to uncover deeper insights.
Analyzing relationships between different data entities.
Finding correlations between linked data points.
Supporting comprehensive cross-table analysis.
Enabling more complex business intelligence.
Revealing hidden dependencies in the data structure.
Facilitating holistic data interpretation and decision making.""",
                "type": "join",
                "generated_at": timestamp,
                "focus_areas": current_focus
            })
            
        return fallback_questions

def generate_questions(df, table_name, join_relationships=None):
    """Generate relevant questions based on the data"""
    # Get column information
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Generate timestamp for uniqueness
    timestamp = int(time.time() * 1000)
    
    # Random focus areas for context
    analysis_focuses = [
        "trends and patterns", "comparative analysis", "statistical insights",
        "performance metrics", "distribution analysis", "relationship studies"
    ]
    current_focus = random.sample(analysis_focuses, 3)
    
    # Basic questions template
    basic_questions = [
        {
            "question": f"Show the overall distribution of {random.choice(numeric_cols)}" if numeric_cols else "Show basic statistics",
            "visualization": "histogram",
            "purpose": """Provides fundamental understanding of data distribution.
Helps identify the central tendency and spread of values.
Shows potential outliers and unusual patterns.
Supports basic statistical analysis and decision making.
Useful for data quality assessment and validation.
Helps in setting reasonable thresholds and benchmarks.""",
            "type": "read"
        },
        {
            "question": f"Compare {random.choice(numeric_cols)} across different {random.choice(categorical_cols)}" if (numeric_cols and categorical_cols) else "Show category breakdown",
            "visualization": "bar",
            "purpose": """Reveals key differences between categories.
Identifies top and bottom performing segments.
Helps spot trends and patterns across groups.
Supports strategic decision-making processes.
Useful for resource allocation planning.
Highlights areas needing attention or improvement.""",
            "type": "read"
        },
        {
            "question": f"Show the trend of {random.choice(numeric_cols)} over time" if (numeric_cols and date_cols) else "Show temporal patterns",
            "visualization": "line",
            "purpose": """Visualizes temporal patterns and trends.
Helps identify seasonal variations and cycles.
Shows long-term growth or decline patterns.
Supports forecasting and planning activities.
Reveals important time-based insights.
Useful for performance tracking over time.""",
            "type": "read"
        }
    ]

    # Heatmap-specific questions
    heatmap_questions = [
        {
            "question": f"Create a correlation heatmap of all numeric variables" if len(numeric_cols) > 1 else "Show basic correlations",
            "visualization": "heatmap",
            "purpose": """Reveals relationships between multiple numeric variables.
Identifies strong positive and negative correlations.
Helps discover hidden patterns in the data.
Supports feature selection for analysis.
Guides deeper statistical investigation.
Useful for understanding data interdependencies.""",
            "type": "read"
        },
        {
            "question": f"Show a heatmap of {random.choice(numeric_cols)} by {random.choice(categorical_cols)}" if (numeric_cols and categorical_cols) else "Show category patterns",
            "visualization": "heatmap",
            "purpose": """Visualizes patterns across different categories.
Highlights concentration of values in specific segments.
Reveals clustering patterns in the data.
Supports identification of key relationships.
Helps in strategic decision making.
Useful for targeted analysis and planning.""",
            "type": "read"
        }
    ]

    # JOIN-specific questions
    join_questions = []
    if join_relationships:
        for rel in join_relationships[:2]:  # Take up to 2 relationships to add questions
            join_questions.append({
                "question": f"Join {rel['table1']} and {rel['table2']} on {rel['key']} and analyze the combined data",
                "visualization": "table",
                "purpose": """Connects related data from different tables.
Allows for cross-table analysis and insights.
Reveals relationships between different data entities.
Enables more comprehensive data exploration.
Helps identify patterns across related datasets.
Supports integrated business intelligence.
Provides holistic view of the data ecosystem.""",
                "type": "join"
            })
            
            # Add a specific analysis question using JOIN if we have numeric columns
            if numeric_cols:
                join_questions.append({
                    "question": f"Analyze {random.choice(numeric_cols)} from {rel['table1']} joined with {rel['table2']} grouped by {random.choice(categorical_cols) if categorical_cols else rel['key']}",
                    "visualization": "bar",
                    "purpose": """Enables cross-table statistical analysis.
Highlights relationships between different data entities.
Reveals insights that single table analysis might miss.
Supports detailed understanding of data relationships.
Helps track metrics across related data points.
Provides deeper analytical context.
Enhances decision-making with integrated data views.""",
                    "type": "join"
                })

    # Combine all questions and randomly select a subset
    all_questions = basic_questions + heatmap_questions + join_questions
    random.shuffle(all_questions)  # Randomize the order
    
    # Select questions based on the available types
    selected_questions = []
    
    # Always include at least one JOIN question if available
    join_q = [q for q in all_questions if q["type"] == "join"]
    if join_q:
        selected_questions.append(join_q[0])
        all_questions.remove(join_q[0])
    
    # Fill the rest with a mix of other questions
    remaining_count = min(4, len(all_questions))
    selected_questions.extend(random.sample(all_questions, remaining_count))
    
    # Add metadata
    for q in selected_questions:
        q['generated_at'] = timestamp
        q['focus_areas'] = current_focus
    
    return selected_questions

def format_questions(questions, metadata):
    """Format questions with additional context and visualization suggestions"""
    formatted_output = []
    
    for q in questions:
        formatted_q = {
            "question": q["question"],
            "visualizations": q.get("viz_suggestion", "table").split(","),
            "purpose": q.get("purpose", "General data exploration"),
            "context": {
                "requires_numeric": "heatmap" in q.get("viz_suggestion", ""),
                "requires_categorical": any(viz in q.get("viz_suggestion", "") for viz in ["pie", "bar"]),
                "statistical_note": "Correlation analysis available" if metadata["correlation_analysis_possible"] else "Limited correlation analysis possible"
            }
        }
        formatted_output.append(formatted_q)
    
    return formatted_output

def main():
    try:
        # Check if database exists
        check_database_exists()
        
        # Connect to the database
        conn = sqlite3.connect('data.db')
        print("Successfully connected to database.")
        
        # Get all table names
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        tables = [t[0] for t in tables]
        
        if not tables:
            print("No tables found in the database.")
            return
        
        print(f"Found tables: {', '.join(tables)}")
        
        # Detect potential join relationships
        join_relationships, table_columns = detect_join_relationships(conn)
        
        if join_relationships:
            print("\nDetected potential JOIN relationships:")
            for rel in join_relationships:
                print(f"- {rel['table1']} can be joined with {rel['table2']} on {rel['key']}")
        
        # Read the first table for basic questions
        table_name = tables[0]
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        print(f"Successfully read {len(df)} rows from {table_name}.")
        
        # Get suggested questions including JOIN questions if relationships exist
        questions = get_suggested_questions(df, table_name, join_relationships=join_relationships)
        
        # Display questions
        display_questions(questions)
        
        conn.close()
        print("\nDatabase connection closed.")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 