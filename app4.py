from dotenv import load_dotenv
import streamlit as st
import os
import pandas as pd
import google.generativeai as genai

# Load environment variables
load_dotenv(override=True)

# Configure your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini model and provide guidance for pandas code generation
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-pro')  # Ensure 'gemini-pro' is a valid model
    combined_input = f"{prompt}\n{question}"  # Combine as a single string
    response = model.generate_content(combined_input)  # Ensure proper API call
    print(response)  # Optional: log for debugging
    return response.text

# Streamlit app
st.set_page_config(page_title="Data Analysis with Pandas")
st.header("Upload Excel or CSV and Ask Questions")

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
    st.write(df.columns)
    st.dataframe(df)

    # Input for user question
    question = st.text_input("Input your question about the data: ", key="input")

    # Button to submit the question
    submit = st.button("Ask the question")

    # If submit is clicked
    if submit:
        # Create a prompt for Gemini API to generate pandas-based analysis code
        prompt = f"""
        You are an expert in converting English questions into code that manipulates dataframes using pandas.
        The provided data has the following columns: {', '.join(df.columns)}.
        The table is saved as a Pandas dataframe 'df'.
        Please answer the following question by providing pandas code to analyze the data.
        For example, when asked how many unique customers are there in data, the response should be print(df['Customer_Name'].nunique())
        You should not generate any SQL or database-specific code.
        Dont write the word python within the code.
        """

        response = get_gemini_response(question, prompt)

        # Display the generated pandas code
        st.subheader("Generated pandas code:")
        st.code(response, language='python')

        try:
            # Execute the generated code using eval (Note: Be cautious with eval in production)
            # Ensure the code is safe and makes sense for pandas
            local_vars = {'df': df}
            result= exec(response, {}, local_vars)  # Execute the pandas code

            # Fetch the result from local_vars, typically the result is stored in a variable by the code
            result_df = local_vars.get('result', df)  # Assume the result is stored in 'result'
            st.subheader("The response is:")
            st.subheader(result)

        except Exception as e:
            st.error(f"Error executing pandas code: {e}")
