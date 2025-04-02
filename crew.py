import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
from pathlib import Path
import sqlite3
import os
from litellm import completion
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Create directory for saving graphs
GRAPH_DIR = "generated_graphs"
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

def parse_visualization_request(natural_query: str, table_name: str) -> dict:
    """Parse natural language query into visualization parameters using LLM"""
    # Get table schema
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    schema_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    schema = cursor.execute(schema_query).fetchone()[0]
    
    # Get sample data
    sample_query = f"SELECT * FROM {table_name} LIMIT 5"
    df = pd.read_sql_query(sample_query, conn)
    columns = df.columns.tolist()
    
    # Create context for LLM
    context = f"""
    Table name: {table_name}
    Available columns: {', '.join(columns)}
    Sample data preview:
    {df.to_string()}
    
    Schema:
    {schema}
    """
    
    system_prompt = """You are a data visualization expert. Analyze the natural language request and the table 
    structure to determine the most appropriate visualization type and columns to use. Return a JSON object with 
    the following structure based on the visualization type:

    For bar charts:
    {
        "visualization_type": "bar_chart",
        "x_column": "column_name",
        "y_column": "column_name",
        "title": "chart_title"
    }

    For scatter plots:
    {
        "visualization_type": "scatter_plot",
        "x_column": "numeric_column",
        "y_column": "numeric_column",
        "title": "chart_title"
    }

    For pie charts:
    {
        "visualization_type": "pie_chart",
        "category_column": "category_column",
        "value_column": "numeric_column",
        "title": "chart_title"
    }

    For time series:
    {
        "visualization_type": "time_series",
        "time_column": "date_column",
        "value_column": "numeric_column",
        "title": "chart_title"
    }

    Choose the most appropriate visualization type based on the data and request.
    Only use columns that exist in the table.
    """

    message = f"""
    Context:
    {context}
    
    Natural language request:
    {natural_query}
    
    Generate the visualization parameters as a JSON object.
    """

    response = completion(
        model="groq/llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    )

    # Extract JSON from the response
    try:
        # Try to find JSON-like content in the response
        content = response.choices[0]['message']['content']
        # Find content between curly braces
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            params = json.loads(json_str)
            params['table_name'] = table_name
            return params
    except Exception as e:
        raise Exception(f"Failed to parse visualization parameters: {str(e)}")

def create_bar_chart(data_info: str) -> str:
    """Creates a bar chart from the SQL data. Input should contain table_name and columns to plot."""
    try:
        # Parse the input data
        import json
        data_dict = json.loads(data_info)
        table_name = data_dict['table_name']
        x_column = data_dict['x_column']
        y_column = data_dict['y_column']
        title = data_dict.get('title', f'Bar Chart of {y_column} by {x_column}')

        # Connect to the database and get data
        conn = sqlite3.connect('data.db')
        query = f"SELECT {x_column}, {y_column} FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Create the visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x=x_column, y=y_column)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        filename = f"bar_chart_{x_column}_{y_column}.png"
        filepath = os.path.join(GRAPH_DIR, filename)
        plt.savefig(filepath)
        plt.close()

        return f"Bar chart has been saved as {filename} in the {GRAPH_DIR} directory"
    except Exception as e:
        return f"Error creating bar chart: {str(e)}"

def create_scatter_plot(data_info: str) -> str:
    """Creates a scatter plot from the SQL data. Input should contain table_name and columns to plot."""
    try:
        # Parse the input data
        import json
        data_dict = json.loads(data_info)
        x_column = data_dict['x_column']
        y_column = data_dict['y_column']
        title = data_dict.get('title', f'Scatter Plot of {y_column} vs {x_column}')

        # Connect to the database and get data
        conn = sqlite3.connect('data.db')
        
        # Handle join query if present
        if 'join_query' in data_dict and data_dict['join_query']:
            query = f"SELECT {x_column}, {y_column} FROM ({data_dict['join_query']})"
        else:
            table_name = data_dict['table_name']
            query = f"SELECT {x_column}, {y_column} FROM {table_name}"
            
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Create the visualization
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_column, y=y_column)
        plt.title(title)
        plt.tight_layout()

        # Save the plot
        filename = f"scatter_plot_{x_column}_{y_column}.png"
        filepath = os.path.join(GRAPH_DIR, filename)
        plt.savefig(filepath)
        plt.close()

        return f"Scatter plot has been saved as {filename} in the {GRAPH_DIR} directory"
    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"

def create_pie_chart(data_info: str) -> str:
    """Creates a pie chart from the SQL data. Input should contain table_name and columns to plot."""
    try:
        # Parse the input data
        import json
        data_dict = json.loads(data_info)
        category_column = data_dict['category_column']
        value_column = data_dict['value_column']
        title = data_dict.get('title', f'Pie Chart of {value_column} by {category_column}')

        # Connect to the database and get data
        conn = sqlite3.connect('data.db')
        
        # Handle join query if present
        if 'join_query' in data_dict and data_dict['join_query']:
            query = f"SELECT {category_column}, SUM({value_column}) as total FROM ({data_dict['join_query']}) GROUP BY {category_column}"
        else:
            table_name = data_dict['table_name']
            query = f"SELECT {category_column}, SUM({value_column}) as total FROM {table_name} GROUP BY {category_column}"
            
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Create the visualization
        plt.figure(figsize=(10, 8))
        plt.pie(df['total'], labels=df[category_column], autopct='%1.1f%%')
        plt.title(title)
        plt.axis('equal')

        # Save the plot
        filename = f"pie_chart_{category_column}_{value_column}.png"
        filepath = os.path.join(GRAPH_DIR, filename)
        plt.savefig(filepath)
        plt.close()

        return f"Pie chart has been saved as {filename} in the {GRAPH_DIR} directory"
    except Exception as e:
        return f"Error creating pie chart: {str(e)}"

def create_time_series(data_info: str) -> str:
    """Creates a time series plot from the SQL data. Input should contain table_name and columns to plot."""
    try:
        # Parse the input data
        import json
        data_dict = json.loads(data_info)
        time_column = data_dict['time_column']
        value_column = data_dict['value_column']
        title = data_dict.get('title', f'Time Series of {value_column} over {time_column}')

        # Connect to the database and get data
        conn = sqlite3.connect('data.db')
        
        # Handle join query if present
        if 'join_query' in data_dict and data_dict['join_query']:
            query = f"SELECT {time_column}, {value_column} FROM ({data_dict['join_query']}) ORDER BY {time_column}"
        else:
            table_name = data_dict['table_name']
            query = f"SELECT {time_column}, {value_column} FROM {table_name} ORDER BY {time_column}"
            
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Create the visualization
        plt.figure(figsize=(12, 6))
        plt.plot(df[time_column], df[value_column], marker='o')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        filename = f"time_series_{value_column}.png"
        filepath = os.path.join(GRAPH_DIR, filename)
        plt.savefig(filepath)
        plt.close()

        return f"Time series plot has been saved as {filename} in the {GRAPH_DIR} directory"
    except Exception as e:
        return f"Error creating time series plot: {str(e)}"

class HeatmapCreator:
    def __init__(self, df=None, join_query=None):
        if df is not None:
            self.df = df
        elif join_query is not None:
            # Load data from join query
            conn = sqlite3.connect('data.db')
            self.df = pd.read_sql_query(join_query, conn)
            conn.close()
        else:
            raise ValueError("Either df or join_query must be provided")
            
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns

    def create_correlation_heatmap(self):
        """Create an interactive correlation heatmap"""
        if len(self.numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns to create a heatmap")
            return None

        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()

        # Create heatmap using plotly
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            hoverongaps=False,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Correlation: %{z:.2f}<extra></extra>"
        ))

        # Update layout
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=800
        )

        return fig

    def create_advanced_heatmap(self):
        """Create an advanced heatmap with additional statistical information"""
        if len(self.numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns to create a heatmap")
            return None

        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()

        # Calculate p-values
        p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                              index=corr_matrix.index, 
                              columns=corr_matrix.columns)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i != j:
                    stat, p = stats.pearsonr(self.df[corr_matrix.columns[i]], 
                                           self.df[corr_matrix.columns[j]])
                    p_values.iloc[i,j] = p

        # Create annotation text
        annotations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                annotations.append(
                    dict(
                        x=corr_matrix.columns[i],
                        y=corr_matrix.columns[j],
                        text=f"r={corr_matrix.iloc[i,j]:.2f}\np={p_values.iloc[i,j]:.3f}",
                        showarrow=False,
                        font=dict(
                            color="white" if abs(corr_matrix.iloc[i,j]) > 0.5 else "black"
                        )
                    )
                )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            hoverongaps=False,
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Correlation: %{z:.2f}<extra></extra>"
        ))

        # Update layout with annotations
        fig.update_layout(
            title="Advanced Correlation Heatmap",
            annotations=annotations,
            width=900,
            height=900,
            xaxis_title="Features",
            yaxis_title="Features"
        )

        return fig

    def create_categorical_heatmap(self, target_col=None):
        """Create a heatmap showing relationships between categorical variables"""
        if not target_col:
            if len(self.categorical_cols) > 0:
                target_col = self.categorical_cols[0]
            else:
                st.warning("No categorical columns found")
                return None

        # Calculate cramers V correlation for categorical variables
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            min_dim = min(confusion_matrix.shape) - 1
            return np.sqrt(chi2 / (n * min_dim))

        # Create correlation matrix for categorical variables
        cat_corr = pd.DataFrame(index=self.categorical_cols, columns=self.categorical_cols)
        for i in self.categorical_cols:
            for j in self.categorical_cols:
                cat_corr.loc[i,j] = cramers_v(self.df[i], self.df[j])

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cat_corr,
            x=cat_corr.columns,
            y=cat_corr.index,
            colorscale='Viridis',
            hoverongaps=False,
            text=np.round(cat_corr, 2),
            texttemplate='%{text}',
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Cramer's V: %{z:.2f}<extra></extra>"
        ))

        fig.update_layout(
            title="Categorical Variables Association (Cramer's V)",
            xaxis_title="Categories",
            yaxis_title="Categories",
            width=800,
            height=800
        )

        return fig

def display_heatmap_analysis(df=None, join_query=None):
    """Main function to display heatmap analysis"""
    st.title("📊 Heatmap Analysis")
    
    try:
        if df is None and join_query is not None:
            # Load data from join query
            conn = sqlite3.connect('data.db')
            df = pd.read_sql_query(join_query, conn)
            conn.close()
            st.write(f"Analyzing joined data with {len(df)} rows")
        elif df is None:
            st.error("No data provided for heatmap analysis")
            return
            
        heatmap_creator = HeatmapCreator(df)
        
        # Heatmap type selection
        heatmap_type = st.radio(
            "Select Heatmap Type:",
            ["Basic Correlation", "Advanced Statistical", "Categorical Associations"],
            horizontal=True
        )
        
        if heatmap_type == "Basic Correlation":
            fig = heatmap_creator.create_correlation_heatmap()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add correlation interpretation
                st.write("### Interpretation Guide:")
                st.write("""
                - 1.0: Perfect positive correlation
                - 0.7 to 1.0: Strong positive correlation
                - 0.3 to 0.7: Moderate positive correlation
                - 0 to 0.3: Weak positive correlation
                - 0: No correlation
                - -0.3 to 0: Weak negative correlation
                - -0.7 to -0.3: Moderate negative correlation
                - -1.0 to -0.7: Strong negative correlation
                """)
                
        elif heatmap_type == "Advanced Statistical":
            fig = heatmap_creator.create_advanced_heatmap()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistical interpretation
                st.write("### Statistical Interpretation:")
                st.write("""
                - r: Correlation coefficient
                - p: p-value (statistical significance)
                - p < 0.05: Statistically significant correlation
                - p ≥ 0.05: Not statistically significant
                """)
                
        else:  # Categorical Associations
            if len(heatmap_creator.categorical_cols) > 0:
                target_col = st.selectbox(
                    "Select Target Variable:",
                    heatmap_creator.categorical_cols
                )
                fig = heatmap_creator.create_categorical_heatmap(target_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add Cramer's V interpretation
                    st.write("### Cramer's V Interpretation:")
                    st.write("""
                    - 0.0 to 0.1: Negligible association
                    - 0.1 to 0.3: Weak association
                    - 0.3 to 0.5: Moderate association
                    - 0.5 to 0.8: Strong association
                    - 0.8 to 1.0: Very strong association
                    """)
            else:
                st.warning("No categorical columns found in the dataset")

        # Add download options
        if st.button("Download Correlation Matrix"):
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            csv = corr_matrix.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error in heatmap analysis: {str(e)}")

class VisualizationCrew:
    def __init__(self):
        self.llm = llm
        self.tools = [create_bar_chart, create_scatter_plot, create_pie_chart, create_time_series]
        self.agent = Agent(
            role="Data Visualization Expert",
            goal="Create insightful visualizations from SQL data",
            backstory="""You are an expert in data visualization. Your job is to analyze natural language requests 
            and create appropriate visualizations. You understand which type of chart works best for different kinds 
            of data and relationships, and can interpret user requests to create the most meaningful visualizations.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True
        )

    def process_request(self, natural_query: str, table_name: str) -> str:
        """Process a natural language visualization request"""
        try:
            # Parse the natural language request into visualization parameters
            viz_params = parse_visualization_request(natural_query, table_name)
            
            # Create the visualization based on the type
            viz_type = viz_params['visualization_type']
            if viz_type == 'bar_chart':
                return create_bar_chart(json.dumps(viz_params))
            elif viz_type == 'scatter_plot':
                return create_scatter_plot(json.dumps(viz_params))
            elif viz_type == 'pie_chart':
                return create_pie_chart(json.dumps(viz_params))
            elif viz_type == 'time_series':
                return create_time_series(json.dumps(viz_params))
            else:
                return f"Unsupported visualization type: {viz_type}"
                
        except Exception as e:
            return f"Error processing visualization request: {str(e)}"

    def crew(self, natural_query: str, table_name: str):
        task = Task(
            description=f"Create a visualization based on the request: {natural_query}",
            agent=self.agent,
            expected_output="A detailed response about the visualization created and its location."
        )
        
        # Add the process_request method to the agent's context
        self.agent.process_request = lambda q: self.process_request(q, table_name)
        
        return Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True
        )

# Example usage
if __name__ == "__main__":
    # Example natural language request
    request = "Show me a bar chart of total sales by category"
    table_name = "sales_data"
    
    # Create and execute the crew
    viz_crew = VisualizationCrew()
    result = viz_crew.crew(natural_query=request, table_name=table_name).kickoff()
    print("\nVisualization Result:")
    print(result)

    # For testing purposes
    sample_df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.choice(['X', 'Y', 'Z'], 100),
        'D': np.random.choice(['P', 'Q'], 100)
    })
    display_heatmap_analysis(sample_df)