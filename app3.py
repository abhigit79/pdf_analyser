from dotenv import load_dotenv 

load_dotenv(override=True)

import streamlit as st
import os
import sqlite3
import pandas as pd
import google.generativeai as genai

# Configure your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini model and provide SQL query
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-pro')
    combined_input = f"{prompt}\n{question}"  # Combine as a single string
    response = model.generate_content(combined_input)
    print(response)
    return response.text

# Function to retrieve query results from SQL database
def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    return rows

# Streamlit app
st.set_page_config(page_title="SQL Query Generator from Excel/CSV")
st.header("Upload Excel or CSV and Ask SQL Questions")

# Upload file
uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xls", "xlsx", "csv"])
if uploaded_file is not None:
    # Check file type and read the file accordingly
    if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Unsupported file type.")

    # Display the dataframe in the app
    st.write("Data from the uploaded file:")
    st.dataframe(df)

    # Input for user question
    question = st.text_input("Input your question about the data: ", key="input")

    # Button to submit the question
    submit = st.button("Ask the question")

    # If submit is clicked
    if submit:
        prompt = f"""
        You are an expert in converting English questions to SQL queries!
        The provided data has the following columns: {', '.join(df.columns)}.
        The name of the table is customer.
        Please answer the following question in SQL format.
        Don't include the text 'SQL' in the beginning or end.
        """
        response = get_gemini_response(question, prompt)

        # Display the generated SQL query
        st.subheader("Generated SQL Query:")
        st.code(response, language='sql')

        # Execute the generated SQL query on the dataframe
        try:
            result_df = pd.read_sql_query(response, sqlite3.connect("customer.db"))
            st.subheader("The response is:")
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"Error executing SQL query: {e}")
