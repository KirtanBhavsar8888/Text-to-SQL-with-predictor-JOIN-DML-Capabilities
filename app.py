import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import numpy as np
from litellm import completion
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from question_generator import get_suggested_questions
import atexit
from crew import display_heatmap_analysis
from predictor import prediction_page

# Set up the API key
os.environ['GROQ_API_KEY'] = "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"

# Initialize session state variables
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'table_name' not in st.session_state:
    st.session_state.table_name = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_query' not in st.session_state:
    st.session_state.current_query = ''
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = None
if 'db_path' not in st.session_state:
    st.session_state.db_path = None
if 'dml_enabled' not in st.session_state:
    st.session_state.dml_enabled = False
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []

def extract_sql_from_markdown(text):
    """Extract SQL query from markdown code blocks"""
    # Pattern to match code blocks with or without language specifier
    code_pattern = r"```.*?\n(.*?)```"
    matches = re.findall(code_pattern, text, re.DOTALL)
    
    if matches:
        # Return the first code block found, removing any leading/trailing whitespace
        return matches[0].strip()
    else:
        # If no code block found, return the text as is
        return text.strip()

def generate_db_context(df, table_name):
    """Generate a comprehensive context about the database"""
    context = f"Database Table Name: {table_name}\n\n"
    
    # Add column information
    context += "Columns:\n"
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        sample = ', '.join(map(str, df[col].dropna().unique()[:3]))
        context += f"- {col} (Type: {dtype}, Unique Values: {nunique}, Sample Values: {sample})\n"
    
    # Add basic statistics
    context += f"\nTotal Records: {len(df)}\n"
    
    # Add numerical column statistics
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        context += "\nNumerical Columns Statistics:\n"
        stats = df[num_cols].describe()
        for col in num_cols:
            context += f"- {col}: min={stats[col]['min']:.2f}, max={stats[col]['max']:.2f}, mean={stats[col]['mean']:.2f}\n"
    
    # Add categorical column information
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        context += "\nCategorical Columns Top Values:\n"
        for col in cat_cols:
            top_vals = df[col].value_counts().head(3)
            context += f"- {col}: {', '.join(map(str, top_vals.index))}\n"
    
    return context

def detect_tables_in_query(natural_query):
    """Detect potential table names in a natural language query"""
    # Get all tables from the database
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    all_tables = [table[0] for table in cursor.fetchall()]
    conn.close()
    
    # Look for table names in the query
    mentioned_tables = []
    natural_query_lower = natural_query.lower()
    
    for table in all_tables:
        if table.lower() in natural_query_lower:
            mentioned_tables.append(table)
    
    # If no tables explicitly mentioned, return all tables
    if not mentioned_tables:
        return all_tables
    
    return mentioned_tables

def generate_multi_table_context(table_names):
    """Generate database context for multiple tables"""
    context = "Database Tables:\n\n"
    table_contexts = {}
    table_schemas = {}
    join_relationships = []
    
    conn = sqlite3.connect('data.db')
    
    for table in table_names:
        df = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", conn)
        
        # Get table schema
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        schema = f"Table: {table}\n"
        schema += "Columns:\n"
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            schema += f"- {col_name} ({col_type})\n"
        
        table_schemas[table] = schema
        
        # Generate single table context
        table_context = generate_db_context(df, table)
        table_contexts[table] = table_context
    
    # Detect potential join relationships
    for i, table1 in enumerate(table_names):
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table1})")
        cols1 = [col[1] for col in cursor.fetchall()]
        
        for j, table2 in enumerate(table_names):
            if i >= j:  # Skip same table and avoid duplicates
                continue
                
            cursor.execute(f"PRAGMA table_info({table2})")
            cols2 = [col[1] for col in cursor.fetchall()]
            
            # Find common columns - potential join keys
            common_cols = set(cols1) & set(cols2)
            
            # Look for potential ID fields
            id_columns = [col for col in common_cols if 
                          col.lower().endswith('id') or 
                          col.lower() == 'id' or
                          col.lower() == table1.lower() + '_id' or
                          col.lower() == table2.lower() + '_id']
            
            if id_columns:
                for col in id_columns:
                    join_relationships.append({
                        'table1': table1,
                        'table2': table2,
                        'key': col,
                        'relationship': f"{table1}.{col} = {table2}.{col}"
                    })
            elif common_cols:
                # Use first common column if no ID columns found
                col = list(common_cols)[0]
                join_relationships.append({
                    'table1': table1,
                    'table2': table2,
                    'key': col,
                    'relationship': f"{table1}.{col} = {table2}.{col}"
                })
    
    conn.close()
    
    # Combine all contexts
    for table, schema in table_schemas.items():
        context += f"{schema}\n"
    
    context += "\nTable Relationships:\n"
    if join_relationships:
        for rel in join_relationships:
            context += f"- {rel['table1']} can be joined with {rel['table2']} on {rel['relationship']}\n"
    else:
        context += "- No clear relationships detected between tables\n"
    
    # Add sample data for each table
    context += "\nSample Data:\n"
    for table, table_context in table_contexts.items():
        sample_data = table_context.split("Total Records:")[0]  # Get just the column info part
        context += f"\n{sample_data}\n"
    
    return context, join_relationships

def generate_sql_query(natural_query, db_context, multi_table=False):
    """Generate SQL query from natural language using LiteLLM"""
    try:
        # First check if this might be a JOIN query
        if multi_table:
            # Detect mentioned tables
            tables = detect_tables_in_query(natural_query)
            
            # If multiple tables detected or JOIN keywords present, generate multi-table context
            if len(tables) > 1 or any(keyword in natural_query.lower() for keyword in ['join', 'combine', 'merge', 'relate', 'linked', 'connecting']):
                multi_context, join_relationships = generate_multi_table_context(tables)
                
                # Enhanced system prompt for JOIN queries
                system_prompt = """You are an expert SQL query generator specializing in JOIN operations. 
                Your task is to convert natural language questions into SQL queries, potentially using JOIN operations.
                Use the provided database context to generate accurate SQL queries. Return the SQL query wrapped in code blocks using three backticks.
                
                Make sure to:
                1. Use the correct table names in the FROM and JOIN clauses
                2. Reference only columns that exist in the tables
                3. Use appropriate JOIN type (INNER JOIN, LEFT JOIN, etc.) based on the question
                4. Use proper table alias when necessary to avoid ambiguity
                5. Include ON conditions that accurately represent table relationships
                6. Format the SQL query properly with appropriate indentation
                7. Return ONLY the executable SQL query without any explanations
                
                For join queries, follow this structure:
                ```
                SELECT [columns]
                FROM table1
                JOIN table2 ON table1.key = table2.key
                [WHERE conditions]
                [GROUP BY columns]
                [ORDER BY columns]
                ```
                """
                
                message = f"""Database Context:
                {multi_context}
                
                Join Relationships Available:
                {[(rel['table1'], rel['table2'], rel['key']) for rel in join_relationships]}
                
                Natural Language Query: {natural_query}
                
                Generate a SQL query that answers this question. If the query requires data from multiple tables, 
                use appropriate JOIN operations. Return ONLY the SQL query wrapped in code blocks using three backticks."""
                
                response = completion(
                    model="groq/llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                # Extract SQL from markdown code blocks
                sql_query = extract_sql_from_markdown(response.choices[0].message.content)
                return sql_query, join_relationships
        
        # Default single-table query handling
        system_prompt = """You are an expert SQL query generator. Your task is to convert natural language questions into SQL queries.
        Use the provided database context to generate accurate SQL queries. Return the SQL query wrapped in code blocks using three backticks.
        Make sure to:
        1. Use the correct table name
        2. Reference only columns that exist in the database
        3. Use appropriate SQL functions and operators
        4. Return a valid, executable SQL query wrapped in code blocks
        5. Format the SQL query properly with appropriate indentation
        """
        
        message = f"""Database Context:
        {db_context}
        
        Natural Language Query: {natural_query}
        
        Generate a SQL query that answers this question. Return the SQL query wrapped in code blocks using three backticks."""
        
        response = completion(
            model="groq/llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract SQL from markdown code blocks
        sql_query = extract_sql_from_markdown(response.choices[0].message.content)
        return sql_query, None
        
    except Exception as e:
        print(f"Error in generate_sql_query: {str(e)}")
        return None, None

def suggest_visualization(query, data_description, result_df):
    """Use regex patterns to suggest appropriate visualization type"""
    # Convert query to lowercase for better matching
    query = query.lower()
    
    # Define regex patterns for different visualization types
    patterns = {
        'pie': r'percentage|proportion|distribution|share|ratio|breakdown|composition',
        'bar': r'compare|comparison|difference|ranking|highest|lowest|top|bottom',
        'line': r'trend|over time|change|growth|decline|progression|time series',
        'scatter': r'correlation|relationship|versus|vs|against|between.*and',
        'histogram': r'distribution|frequency|spread|range',
        'box': r'quartile|distribution|spread|outliers|range|median'
    }
    
    # Check number of columns and their types
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    categorical_cols = result_df.select_dtypes(include=['object']).columns
    
    # Match patterns in the query
    for viz_type, pattern in patterns.items():
        if re.search(pattern, query):
            # Additional checks for data compatibility
            if viz_type == 'pie' and len(categorical_cols) > 0 and len(numeric_cols) > 0:
                return 'pie'
            elif viz_type == 'line' and len(numeric_cols) >= 2:
                return 'line'
            elif viz_type == 'scatter' and len(numeric_cols) >= 2:
                return 'scatter'
            elif viz_type in ['bar', 'histogram', 'box']:
                return viz_type
    
    # Default to bar chart if no pattern matches or if incompatible data types
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        return 'bar'
    elif len(numeric_cols) > 0:
        return 'histogram'
    else:
        return 'table'

def create_visualization(viz_type, df):
    """Create visualization based on suggested type"""
    try:
        if len(df) == 0:
            return None
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if viz_type == 'pie':
            # Ensure we have both categorical and numeric columns
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # Group by categorical column and sum numeric column
                grouped_df = df.groupby(categorical_cols[0])[numeric_cols[0]].sum().reset_index()
                fig = px.pie(
                    grouped_df,
                    names=categorical_cols[0],
                    values=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
            else:
                return None
                
        elif viz_type == 'bar':
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                fig = px.bar(
                    df,
                    x=categorical_cols[0],
                    y=numeric_cols[0],
                    title=f"{numeric_cols[0]} by {categorical_cols[0]}"
                )
            else:
                fig = px.bar(df)
                
        elif viz_type == 'line':
            if len(numeric_cols) >= 2:
                fig = px.line(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f"{numeric_cols[1]} vs {numeric_cols[0]}"
                )
            else:
                fig = px.line(df)
                
        elif viz_type == 'scatter':
            if len(numeric_cols) >= 2:
                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f"{numeric_cols[1]} vs {numeric_cols[0]}"
                )
            else:
                fig = px.scatter(df)
                
        elif viz_type == 'histogram':
            if len(numeric_cols) > 0:
                fig = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]}"
                )
            else:
                fig = px.histogram(df)
                
        elif viz_type == 'box':
            if len(numeric_cols) > 0:
                fig = px.box(
                    df,
                    y=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]}"
                )
            else:
                fig = px.box(df)
                
        elif viz_type == 'heatmap':
            display_heatmap_analysis(df)
            
        else:  # Default to table
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns)),
                cells=dict(values=[df[col] for col in df.columns])
            )])
            
        fig.update_layout(template="plotly_white")
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def toggle_dml_mode():
    st.session_state.dml_enabled = not st.session_state.dml_enabled

def validate_query_intent(question: str, dml_enabled: bool) -> tuple[bool, str]:
    """
    Validate if the query intent is read-only or attempting to modify the database.
    Returns (is_valid, message)
    """
    # Convert question to lowercase for easier pattern matching
    question_lower = question.lower()
    
    # List of dangerous keywords that are always blocked for safety
    dangerous_keywords = ['drop', 'truncate', 'delete from', 'alter table', 'create table']
    
    # List of DML keywords that are only allowed when DML is enabled
    dml_keywords = ['insert into', 'update', 'replace into', 'delete', 'modify']
    
    # List of safe keywords that indicate read operations
    safe_keywords = [
        'show', 'display', 'select', 'find', 'count', 'calculate', 'what', 
        'how many', 'average', 'list', 'tell me', 'compare', 'analyze'
    ]
    
    # First check for dangerous operations - always block these
    for keyword in dangerous_keywords:
        if keyword in question_lower:
            return False, f"⚠️ ERROR: Dangerous operations like '{keyword}' are not allowed for security reasons."
    
    # If DML is enabled, allow DML operations
    if dml_enabled:
        # Check for explicit SQL to ensure natural language isn't misinterpreted
        has_explicit_sql = any(f"sql {keyword}" in question_lower or f"{keyword} sql" in question_lower for keyword in dml_keywords)
        
        if has_explicit_sql:
            return True, "🔄 DML operation allowed (DML mode enabled)"
        
        # Allow natural language that suggests modification
        for keyword in dml_keywords:
            if keyword in question_lower:
                return True, "🔄 DML operation allowed (DML mode enabled)"
        
        # Always allow safe keywords
        has_safe_keyword = any(keyword in question_lower for keyword in safe_keywords)
        if has_safe_keyword:
            return True, "✅ Query approved"
        
        return True, "✅ Query approved (DML mode enabled)"
    
    # If DML is disabled, block DML operations
    for keyword in dml_keywords:
        if keyword in question_lower:
            return False, f"⚠️ ERROR: Data modification operations like '{keyword}' are not allowed while DML mode is off. Please enable DML mode for modification operations."
    
    # Ensure at least one safe keyword is present in read-only mode
    has_safe_keyword = any(keyword in question_lower for keyword in safe_keywords)
    if not has_safe_keyword:
        return False, "❓ Please rephrase your question to focus on reading or analyzing the data. Start with words like 'show', 'find', 'calculate', etc."
    
    return True, "✅ Query approved"

def generate_query_explanation(natural_query: str, sql_query: str, db_context: str) -> str:
    """Generate a human-readable explanation of the SQL query"""
    try:
        prompt = f"""
        Given the following:
        1. Natural Language Question: {natural_query}
        2. SQL Query: {sql_query}
        3. Database Context: {db_context}

        Provide a clear explanation in two parts:
        1. What the question is asking for (in simple terms)
        2. How the SQL query works to answer this question (explain each part)

        Keep the explanation concise but informative, using bullet points where appropriate.
        """
        
        response = completion(
            model="groq/llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an expert at explaining SQL queries in simple terms."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def init_session_state():
    if 'uploaded_files_data' not in st.session_state:
        st.session_state.uploaded_files_data = {}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'uploaded_files_list' not in st.session_state:
        st.session_state.uploaded_files_list = []

def generate_detailed_report(df):
    report = {}
    
    # Basic Statistics
    report['basic_stats'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    }
    
    # Column Analysis
    report['column_analysis'] = {}
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'null_percentage': f"{(df[col].isnull().sum() / len(df) * 100):.2f}%",
            'unique_values': df[col].nunique()
        }
        
        # Safely get sample values
        try:
            sample_values = list(df[col].dropna().unique()[:5])
            col_info['sample_values'] = sample_values
        except:
            col_info['sample_values'] = []
        
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                col_info.update({
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                })
            except:
                pass
        elif df[col].dtype == 'object':
            try:
                col_info.update({
                    'top_values': df[col].value_counts().head(5).to_dict()
                })
            except:
                col_info.update({'top_values': {}})
            
        report['column_analysis'][col] = col_info
    
    return report

def analyze_categorical_columns(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    analysis = {}
    
    for col in cat_cols:
        try:
            value_counts = df[col].value_counts()
            unique_count = len(value_counts)
            
            analysis[col] = {
                'unique_values': unique_count,
                'top_categories': value_counts.head(5).to_dict(),
                'null_count': df[col].isnull().sum(),
                'distribution': value_counts.to_dict()
            }
        except:
            analysis[col] = {
                'unique_values': 0,
                'top_categories': {},
                'null_count': 0,
                'distribution': {}
            }
    
    return analysis

def safe_sample_values(df, col):
    try:
        return ', '.join(str(x) for x in df[col].dropna().unique()[:5])
    except:
        return "Unable to display"

def display_column_analysis(df):
    st.write("### 📋 Column Analysis")
    for col in df.columns:
        with st.expander(f"Column: {col}"):
            st.write(f"**Type:** {df[col].dtype}")
            st.write(f"**Unique Values:** {df[col].nunique()}")
            st.write(f"**Missing Values:** {df[col].isnull().sum()} ({(df[col].isnull().sum() / len(df) * 100):.2f}%)")
            st.write(f"**Sample Values:** {safe_sample_values(df, col)}")

def display_numerical_statistics(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        st.write("### 📊 Numerical Column Statistics")
        for col in num_cols:
            with st.expander(f"Numerical Column: {col}"):
                try:
                    stats = df[col].describe()
                    st.write(f"**Mean:** {stats['mean']:.2f}")
                    st.write(f"**Median:** {df[col].median():.2f}")
                    st.write(f"**Std Dev:** {stats['std']:.2f}")
                    st.write(f"**Min:** {stats['min']:.2f}")
                    st.write(f"**Max:** {stats['max']:.2f}")
                    st.write(f"**25%:** {stats['25%']:.2f}")
                    st.write(f"**75%:** {stats['75%']:.2f}")
                    
                    # Distribution plot
                    try:
                        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.write("Could not generate histogram")
                except:
                    st.write("Could not calculate statistics")

def display_categorical_analysis(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        st.write("### 📊 Categorical Column Analysis")
        for col in cat_cols:
            with st.expander(f"Categorical Column: {col}"):
                try:
                    value_counts = df[col].value_counts()
                    st.write(f"**Unique Values:** {len(value_counts)}")
                    st.write(f"**Top Categories:**")
                    for val, count in value_counts.head(5).items():
                        st.write(f"- {val}: {count} ({count/len(df)*100:.2f}%)")
                    
                    # Distribution plot
                    if len(value_counts) > 0:
                        values = value_counts.values[:10]
                        names = value_counts.index[:10]
                        
                        if len(values) > 0 and len(names) > 0:
                            try:
                                fig = px.pie(values=values, names=names, title=f"Top 10 Categories in {col}")
                                st.plotly_chart(fig, use_container_width=True)
                            except:
                                st.write("Could not generate pie chart")
                except:
                    st.write("Could not analyze categorical data")

def display_database_statistics(df, dataset_name):
    st.write("### 📚 Database Statistics")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Column type breakdown
    st.write("#### Column Type Breakdown")
    try:
        dtype_counts = df.dtypes.value_counts().reset_index()
        dtype_counts.columns = ["Data Type", "Count"]
        
        if len(dtype_counts) > 0:
            fig = px.bar(dtype_counts, x="Data Type", y="Count", 
                         title="Column Data Types", color="Data Type")
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.write("Could not generate column type breakdown")
    
    # Missing values
    st.write("#### Missing Values Summary")
    try:
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum(),
            'Percentage': df.isnull().sum() / len(df) * 100
        })
        missing = missing.sort_values('Missing Values', ascending=False)
        
        if missing['Missing Values'].sum() > 0:
            st.dataframe(missing, use_container_width=True)
        else:
            st.write("No missing values in the dataset!")
    except:
        st.write("Could not analyze missing values")

def run_text_to_sql():
    # st.title("📊 Data Analysis Platform")    
    init_session_state()

    # Initialize DML mode in session state
    if 'dml_enabled' not in st.session_state:
        st.session_state.dml_enabled = False

    # Section 1: File Upload
    st.markdown("---")
    st.header("1️⃣ Upload Your Data")
    
    uploaded_files = st.file_uploader(
        "Upload CSV Files",
        type=['csv'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files_list:
                try:
                    df = pd.read_csv(file)
                    
                    # Ask user to name/classify the data
                    default_name = os.path.splitext(file.name)[0]
                    data_name = st.text_input(
                        f"What would you like to call this dataset? ({file.name})",
                        value=default_name,
                        key=f"name_{file.name}"
                    )
                    
                    if st.button(f"Confirm name for {file.name}", key=f"confirm_{file.name}"):
                        st.session_state.uploaded_files_data[data_name] = df
                        st.session_state.uploaded_files_list.append(file.name)
                        st.session_state.analysis_results[data_name] = {
                            'detailed_report': generate_detailed_report(df),
                            'categorical_analysis': analyze_categorical_columns(df)
                        }
                        
                        # Create SQLite database tables
                        conn = sqlite3.connect('data.db')
                        df.to_sql(data_name, conn, if_exists='replace', index=False)
                        conn.close()
                        
                        st.success(f"✅ Successfully loaded and saved to database: {data_name}")
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")

    # Section 2: Data Preview
    if st.session_state.uploaded_files_data:
        st.markdown("---")
        st.header("2️⃣ Data Preview")
        
        # Dataset selector for preview
        preview_dataset = st.selectbox(
            "Select a dataset to preview",
            options=list(st.session_state.uploaded_files_data.keys()),
            key="preview_selector"
        )
        
        if preview_dataset:
            df = st.session_state.uploaded_files_data[preview_dataset]
            st.write(f"### Preview of {preview_dataset}")
            st.dataframe(df.head(), use_container_width=True)
            
            # Basic Info
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df))
            col2.metric("Total Columns", len(df.columns))
            col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            # Column Info
            st.write("### Column Information")
            col_info = pd.DataFrame({
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Section 3: Detailed Analysis
        st.markdown("---")
        st.header("3️⃣ Detailed Analysis")
        
        # Dataset selector for detailed analysis
        analysis_dataset = st.selectbox(
            "Select a dataset for detailed analysis",
            options=list(st.session_state.uploaded_files_data.keys()),
            key="analysis_selector"
        )
        
        if analysis_dataset:
            df = st.session_state.uploaded_files_data[analysis_dataset]
            st.write(f"## Detailed Analysis for: {analysis_dataset}")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Column Analysis", 
                "Numerical Statistics", 
                "Categorical Analysis",
                "Database Statistics"
            ])
            
            with tab1:
                display_column_analysis(df)
                
            with tab2:
                display_numerical_statistics(df)
                
            with tab3:
                display_categorical_analysis(df)
                
            with tab4:
                display_database_statistics(df, analysis_dataset)

        # Section 4: Natural Language Query
        st.markdown("---")
        st.header("4️⃣ Natural Language Query")
        
        if len(st.session_state.uploaded_files_data) > 0:
            # st.subheader("🔍 Natural Language Query")
            st.write("Ask questions about your data in plain English")
            
            # Initialize query dataset and context
            if 'query_dataset' not in st.session_state:
                st.session_state.query_dataset = list(st.session_state.uploaded_files_data.keys())[0]
            if 'suggested_questions' not in st.session_state:
                st.session_state.suggested_questions = None
            if 'selected_viz_types' not in st.session_state:
                st.session_state.selected_viz_types = []
            
            # Query configuration options in a single row
            options_col1, options_col2, options_col3 = st.columns(3)
            
            with options_col1:
                # Multi-table query toggle
                enable_joins = st.checkbox("✨ Multi-table queries", value=True, 
                                         help="Enable to ask questions that combine data from multiple tables")
            
            with options_col2:
                # DML Mode toggle with warning
                dml_toggle = st.checkbox("🔄 Enable DML Mode", value=st.session_state.dml_enabled,
                                        help="Allow data modification operations (INSERT, UPDATE, DELETE)")
                
                # Update session state for DML
                if dml_toggle != st.session_state.dml_enabled:
                    st.session_state.dml_enabled = dml_toggle
                    if dml_toggle:
                        st.warning("⚠️ DML Mode enabled. Your queries can now modify the database.")
                    else:
                        st.success("✅ Read-only mode active. Data is protected from modifications.")
            
            with options_col3:
                # Dataset selector for queries
                query_dataset = st.selectbox(
                    "Primary dataset:",
                    options=list(st.session_state.uploaded_files_data.keys()),
                    key="query_dataset_selector",
                    index=list(st.session_state.uploaded_files_data.keys()).index(st.session_state.query_dataset) 
                        if st.session_state.query_dataset in list(st.session_state.uploaded_files_data.keys()) else 0
                )
            
            st.session_state.query_dataset = query_dataset
            st.session_state.df = st.session_state.uploaded_files_data[query_dataset]
            st.session_state.table_name = query_dataset
            
            # Create two columns for Questions and Visualizations
            main_col1, main_col2 = st.columns(2)

            with main_col1:
                # Suggested Questions Section
                st.write("### 💡 Suggested Questions")
                
                # Generate relationships for JOIN queries if enabled
                join_relationships = None
                if enable_joins and len(st.session_state.uploaded_files_data) > 1:
                    try:
                        table_names = list(st.session_state.uploaded_files_data.keys())
                        _, join_relationships = generate_multi_table_context(table_names)
                    except Exception as e:
                        st.warning(f"Could not detect table relationships: {str(e)}")
                
                if st.button("🔄 Generate New Questions"):
                    try:
                        with st.spinner("Generating fresh questions..."):
                            # Clear previous questions first
                            st.session_state.suggested_questions = None
                            
                            # Generate new questions
                            questions = get_suggested_questions(
                                st.session_state.df, 
                                st.session_state.table_name,
                                dml_enabled=st.session_state.dml_enabled,
                                join_relationships=join_relationships
                            )
                            
                            # Update session state with new questions
                            st.session_state.suggested_questions = questions
                            
                            # Use st.rerun() instead of st.experimental_rerun()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error generating questions: {str(e)}")
                
                # Auto-generate questions if none exist
                if st.session_state.suggested_questions is None:
                    try:
                        with st.spinner("Generating questions..."):
                            # Generate new questions
                            questions = get_suggested_questions(
                                st.session_state.df, 
                                st.session_state.table_name,
                                dml_enabled=st.session_state.dml_enabled,
                                join_relationships=join_relationships
                            )
                            
                            # Update session state with new questions
                            st.session_state.suggested_questions = questions
                    except Exception as e:
                        st.warning(f"Could not generate suggested questions: {str(e)}")
                
                if st.session_state.suggested_questions:
                    for idx, question in enumerate(st.session_state.suggested_questions):
                        with st.container():
                            if st.button(f"📝 Use This Question", key=f"q_{idx}"):
                                st.session_state.current_query = question['question']
                            
                            # Add colored badge for operation type
                            if question.get('type') == 'modify' and st.session_state.dml_enabled:
                                st.markdown(f"<span style='background-color: #FFA500; padding: 2px 6px; border-radius: 10px; color: white; font-size: 0.8em;'>MODIFY</span> **Q{idx+1}:** {question['question']}", unsafe_allow_html=True)
                            elif question.get('type') == 'join':
                                st.markdown(f"<span style='background-color: #4B0082; padding: 2px 6px; border-radius: 10px; color: white; font-size: 0.8em;'>JOIN</span> **Q{idx+1}:** {question['question']}", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='background-color: #008000; padding: 2px 6px; border-radius: 10px; color: white; font-size: 0.8em;'>READ</span> **Q{idx+1}:** {question['question']}", unsafe_allow_html=True)
                            
                            with st.expander("See details"):
                                if question.get('visualization'):
                                    st.write(f"**Suggested Visualization:** {question['visualization']}")
                                if question.get('purpose'):
                                    st.write(f"**Purpose:** {question['purpose']}")
                                if question.get('type'):
                                    operation_type = question.get('type')
                                    if operation_type == 'modify' and not st.session_state.dml_enabled:
                                        st.warning("⚠️ This question requires DML mode to be enabled")
                            st.markdown("---")

            with main_col2:
                # Visualization Selection Section
                st.write("### 📊 Select Visualizations")
                st.write("Choose one or more visualization types:")

                # Create visualization options with multiselect
                viz_options = {
                    'bar': '📊 Bar Chart (for comparisons)',
                    'pie': '🥧 Pie Chart (for proportions)',
                    'scatter': '📍 Scatter Plot (for relationships)',
                    'line': '📈 Line Chart (for trends)',
                    'histogram': '📊 Histogram (for distributions)',
                    'box': '📦 Box Plot (for statistics)',
                    'table': '📋 Table View',
                    'heatmap': '🔥 Heatmap (for correlations)'
                }

                selected_viz_types = st.multiselect(
                    "Select visualization types:",
                    options=list(viz_options.keys()),
                    format_func=lambda x: viz_options[x],
                    default=['bar'],
                    help="You can select multiple visualization types"
                )
                st.session_state.selected_viz_types = selected_viz_types
                
                # Question Input Box
                st.write("### ❓ Your Question")
                query_text = st.text_area(
                    "Enter your question in plain English:",
                    value=st.session_state.current_query,
                    height=80,
                    placeholder="Example: 'What is the average age?' or 'Show sales by region joined with customer data'"
                )
                
                # Show SQL mode hint based on DML status
                if st.session_state.dml_enabled:
                    st.info("🔄 DML mode is enabled. You can use queries like 'Update salary where department is Marketing' or 'Insert new record with name John'")
                else:
                    st.info("🔒 DML mode is disabled. Only SELECT queries are allowed. Enable DML mode to modify data.")
                
                execute_button = st.button("🔍 Execute Query", use_container_width=True)
            
            # Execute query when button is clicked
            if execute_button and query_text:
                st.session_state.current_query = query_text
                
                # Validate query intent
                is_valid, message = validate_query_intent(query_text, st.session_state.dml_enabled)
                if not is_valid:
                    st.error(message)
                else:
                    # Show status message
                    st.info(message)
                    
                    # Generate and execute SQL query
                    with st.spinner("Generating SQL query..."):
                        try:
                            # Prepare context based on mode
                            if enable_joins and len(st.session_state.uploaded_files_data) > 1:
                                table_names = list(st.session_state.uploaded_files_data.keys())
                                multi_context, join_relationships = generate_multi_table_context(table_names)
                                sql_query, _ = generate_sql_query(query_text, multi_context, multi_table=True)
                            else:
                                single_context = generate_db_context(st.session_state.df, st.session_state.table_name)
                                sql_query, _ = generate_sql_query(query_text, single_context)
                                
                            if not sql_query:
                                st.error("Failed to generate SQL query.")
                            else:
                                # Display the generated SQL
                                with st.expander("View Generated SQL Query"):
                                    st.code(sql_query, language="sql")
                                
                                # Execute the query
                                conn = None
                                try:
                                    conn = sqlite3.connect('data.db')
                                    
                                    # Check if this is a SELECT query or a modification query
                                    is_select = sql_query.strip().upper().startswith('SELECT')
                                    
                                    if is_select:
                                        # For SELECT queries, return results
                                        result_df = pd.read_sql_query(sql_query, conn)
                                        
                                        # Display results
                                        st.write("### 📋 Query Results")
                                        st.success(f"Query executed successfully. Found {len(result_df)} results.")
                                        st.dataframe(result_df, use_container_width=True)
                                        
                                        # Only show visualization if we have results
                                        if len(result_df) > 0:
                                            st.write("### 📊 Visualizations")
                                            
                                            # If no visualization selected, auto-suggest one
                                            if not st.session_state.selected_viz_types:
                                                suggested_viz = suggest_visualization(query_text, "", result_df)
                                                st.session_state.selected_viz_types = [suggested_viz]
                                            
                                            # Create tabs for each selected visualization
                                            if len(st.session_state.selected_viz_types) > 0:
                                                viz_tabs = st.tabs([viz_options[viz] for viz in st.session_state.selected_viz_types])
                                                
                                                for i, viz_type in enumerate(st.session_state.selected_viz_types):
                                                    with viz_tabs[i]:
                                                        if viz_type == 'heatmap':
                                                            display_heatmap_analysis(result_df)
                                                        else:
                                                            fig = create_visualization(viz_type, result_df)
                                                            if fig:
                                                                st.plotly_chart(fig, use_container_width=True)
                                                            else:
                                                                st.info(f"Couldn't create {viz_type} visualization with this data")
                                    else:
                                        # For modification queries, execute and show affected rows
                                        cursor = conn.cursor()
                                        cursor.execute(sql_query)
                                        affected_rows = cursor.rowcount
                                        conn.commit()
                                        
                                        st.success(f"🔄 Data modification successful! Affected rows: {affected_rows}")
                                        
                                        # Refresh data in session state after modification
                                        for table_name in st.session_state.uploaded_files_data.keys():
                                            fresh_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                                            st.session_state.uploaded_files_data[table_name] = fresh_df
                                        
                                        # Show updated data preview
                                        st.write("### 📋 Updated Data Preview")
                                        st.dataframe(st.session_state.uploaded_files_data[st.session_state.table_name].head(10), use_container_width=True)
                                    
                                    # Generate explanation
                                    with st.expander("Query Explanation"):
                                        context_to_use = multi_context if enable_joins and len(st.session_state.uploaded_files_data) > 1 else single_context
                                        explanation = generate_query_explanation(query_text, sql_query, context_to_use)
                                        st.write(explanation)
                                
                                    # Download options - only for SELECT queries
                                    if is_select and 'result_df' in locals() and len(result_df) > 0:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            csv = result_df.to_csv(index=False).encode('utf-8')
                                            st.download_button(
                                                "📥 Download Results (CSV)",
                                                data=csv,
                                                file_name="query_results.csv",
                                                mime="text/csv",
                                                use_container_width=True
                                            )
                                        with col2:
                                            # Excel download option
                                            try:
                                                import io
                                                buffer = io.BytesIO()
                                                result_df.to_excel(buffer, index=False)
                                                buffer.seek(0)
                                                st.download_button(
                                                    "📥 Download Results (Excel)",
                                                    data=buffer,
                                                    file_name="query_results.xlsx",
                                                    mime="application/vnd.ms-excel",
                                                    use_container_width=True
                                                )
                                            except Exception as e:
                                                st.write("Excel export not available")
                                except Exception as e:
                                    st.error(f"Error executing SQL query: {str(e)}")
                                finally:
                                    if conn:
                                        conn.close()
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
        
        else:
            st.info("Please upload at least one dataset to use the query feature.")

        # Clear data button
        if st.button("Clear All Data"):
            st.session_state.uploaded_files_data.clear()
            st.session_state.analysis_results.clear()
            st.session_state.uploaded_files_list.clear()
            st.session_state.current_query = ''
            st.session_state.suggested_questions = None
            st.session_state.selected_viz_types = []
            st.session_state.dml_enabled = False
            
            # Also clear the database
            try:
                conn = sqlite3.connect('data.db')
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for table in tables:
                    if table[0] != 'sqlite_sequence':
                        cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
                conn.commit()
                conn.close()
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
                
            st.rerun()
    
    else:
        st.info("👆 Please upload your CSV files to begin analysis")

def run_prediction_engine():
    # Add this function to handle predictor integration
    st.header("🔮 Smart Prediction Engine")
    
    # Map your existing data to predictor's expected format
    if 'uploaded_files_data' in st.session_state:
        st.session_state.uploaded_files = [
            {
                'name': name,
                'df': df,
                'table_name': name
            }
            for name, df in st.session_state.uploaded_files_data.items()
        ]
    
    prediction_page()

def main():
    st.set_page_config(page_title="Data Analysis Platform", layout="wide")
    st.title("📊 Data Analysis Platform")
    
    # Add navigation sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Module",
        ["Text-to-SQL Analysis", "Prediction Engine"],
        index=0
    )
    
    if app_mode == "Text-to-SQL Analysis":
        run_text_to_sql()
    else:
        run_prediction_engine()
    

if __name__ == "__main__":
    main()

# Add this function to clean up temporary files
def cleanup_temp_files():
    try:
        if Path("temp.db").exists():
            os.remove("temp.db")
    except Exception as e:
        print(f"Error cleaning up temporary files: {str(e)}")

# Add cleanup on session end
atexit.register(cleanup_temp_files) 