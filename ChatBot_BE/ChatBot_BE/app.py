from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import json
import re
from dotenv import load_dotenv
import msal
import pyodbc
import platform
import pkg_resources

# Load environment variables
load_dotenv()
print("Environment variables loaded")

# Core configuration
SERVER = os.getenv("SQL_SERVER", "kalpita.database.windows.net")
DATABASE = os.getenv("SQL_DATABASE", "KalpitaRecruit-Dev")
USERNAME = os.getenv("SQL_USERNAME")
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

# Print environment variables for debugging (omitting full API key for security)
print("Environment variables:")
print(f"SQL_SERVER: {SERVER}")
print(f"SQL_DATABASE: {DATABASE}")
print(f"SQL_USERNAME: {'Set' if USERNAME else 'Not set'}")
print(f"TENANT_ID: {'Set' if TENANT_ID else 'Not set'}")
print(f"CLIENT_ID: {'Set' if CLIENT_ID else 'Not set'}")
print(f"AZURE_OPENAI_API_KEY: {'Set' if AZURE_OPENAI_API_KEY else 'Not set'}")
print(f"AZURE_OPENAI_ENDPOINT: {AZURE_OPENAI_ENDPOINT}")
print(f"AZURE_OPENAI_DEPLOYMENT_NAME: {AZURE_OPENAI_DEPLOYMENT_NAME}")
print(f"AZURE_OPENAI_API_VERSION: {AZURE_OPENAI_API_VERSION}")

# Check OpenAI SDK version and configure accordingly
openai_version = pkg_resources.get_distribution("openai").version
is_new_version = openai_version.startswith("1.")
print(f"OpenAI SDK version: {openai_version}")

if is_new_version:
    # For OpenAI SDK v1.x
    from openai import AzureOpenAI
    
    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        print("AzureOpenAI client initialized with SDK v1.x")
    except Exception as e:
        print(f"Error initializing AzureOpenAI client: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    # For OpenAI SDK v0.x
    import openai
    
    try:
        openai.api_key = AZURE_OPENAI_API_KEY
        openai.api_base = AZURE_OPENAI_ENDPOINT
        openai.api_type = "azure"
        openai.api_version = AZURE_OPENAI_API_VERSION
        print("OpenAI configured with SDK v0.x")
    except Exception as e:
        print(f"Error configuring OpenAI v0.x: {str(e)}")
        import traceback
        traceback.print_exc()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

class LoginRequest(BaseModel):
    email: str

class AskLlamaRequest(BaseModel):
    question: str
    userEmail: str
    userRoles: list[dict]
    accessibleTables: list[str]
    format: str = "text"  # Added format parameter with default value "text"

class ConversationalRequest(BaseModel):
    prompt: str
    userEmail: str
    userRoles: list[dict]

class MessageClassificationRequest(BaseModel):
    message: str
    accessibleTables: list[str]
    userEmail: str
    userRoles: list[dict]

def get_access_token():
    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    msal_app = msal.PublicClientApplication(client_id=CLIENT_ID, authority=authority)
    accounts = msal_app.get_accounts(username=USERNAME)
    if accounts:
        result = msal_app.acquire_token_silent(scopes=["https://database.windows.net/.default"], account=accounts[0])
        if result:
            return result["access_token"]
    result = msal_app.acquire_token_interactive(scopes=["https://database.windows.net/.default"])
    if "access_token" in result:
        return result["access_token"]
    raise Exception(f"Failed to acquire token: {result.get('error')} - {result.get('error_description')}")

def establish_connection():
    try:
        driver = "{ODBC Driver 17 for SQL Server}" if platform.system() == 'Windows' else "{ODBC Driver 18 for SQL Server}"
        connection_string = f"Driver={driver};Server={SERVER};Database={DATABASE};Authentication=ActiveDirectoryInteractive;UID={USERNAME};"
        conn = pyodbc.connect(connection_string)
        return conn, None
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def get_all_database_objects():
    """Fetch all tables and views from the connected database"""
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    cursor = conn.cursor()
    
    # Get all tables
    tables_query = """
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_TYPE = 'BASE TABLE'
    """
    cursor.execute(tables_query)
    tables = [row[0] for row in cursor.fetchall()]
    
    # Get all views
    views_query = """
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.VIEWS 
    WHERE TABLE_SCHEMA = 'dbo'
    """
    cursor.execute(views_query)
    views = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return tables + views

def get_role_definitions():
    """Get role definitions from the database"""
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    cursor = conn.cursor()
    
    # Check if the RoleTableMapping table exists
    check_query = """
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'RoleTableMapping'
    """
    cursor.execute(check_query)
    table_exists = cursor.fetchone()[0] > 0
    
    # If RoleTableMapping exists, use it for dynamic RBAC
    if table_exists:
        roles_query = """
        SELECT r.RoleName, rtm.TableName
        FROM [dbo].[RoleTableMapping] rtm
        JOIN [dbo].[Roles] r ON rtm.RoleID = r.RoleID
        WHERE rtm.IsActive = 1
        """
        cursor.execute(roles_query)
        role_mappings = cursor.fetchall()
        
        role_definitions = {}
        for role_name, table_name in role_mappings:
            if role_name not in role_definitions:
                role_definitions[role_name] = []
            role_definitions[role_name].append(table_name)
        
        conn.close()
        return role_definitions
    else:
        conn.close()
        return {
            "Admin": [],
            "Recruiter": ["Sourcing", "Candidate", "Education", "PreferredLocation", "NoticePeriod"],
            "Requestor": ["Request", "Requisition", "Vacancy", "Position", "WorkLocation", "Employee"],
            "Interviewer": ["Feedback", "Interview", "Interviewer"]
        }

def define_table_access_by_role(role_name, all_tables):
    """Define table access based on role and filter from all available tables"""
    role_definitions = get_role_definitions()
    if role_name == "Admin":
        return all_tables
    patterns = role_definitions.get(role_name, [])
    accessible = []
    if not patterns:
        patterns = [role_name]
    for table in all_tables:
        if any(pattern.lower() in table.lower() for pattern in patterns):
            accessible.append(table)
    
    return accessible

def get_user_role(email):
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    cursor = conn.cursor()
    query = """
    SELECT r.RoleID, r.RoleName 
    FROM [dbo].[UserRoleMapping] urm
    JOIN [dbo].[Roles] r ON urm.RoleID = r.RoleID
    WHERE urm.UserEmail = ? AND urm.IsActive = 1
    """
    cursor.execute(query, (email,))
    roles = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
    conn.close()
    if not roles:
        raise HTTPException(status_code=401, detail="No active roles found for this email")
    return roles

def get_table_schema(table_name):
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    cursor = conn.cursor()
    table_check_query = """
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = ?
    """
    cursor.execute(table_check_query, (table_name,))
    table_exists = cursor.fetchone()[0] > 0
    if table_exists:
        schema_query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE, COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """
        cursor.execute(schema_query)
        columns = cursor.fetchall()
        schema = f"CREATE TABLE [dbo].[{table_name}](\n"
        for i, col in enumerate(columns):
            column_name, data_type, max_length, is_nullable, default_value = col
            data_type_str = f"{data_type}({max_length})" if max_length and max_length != -1 and data_type in ('char', 'varchar', 'nchar', 'nvarchar') else data_type
            nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
            default = f" DEFAULT {default_value}" if default_value else ""
            comma = "" if i == len(columns) - 1 else ","
            schema += f"    [{column_name}] [{data_type_str}] {nullable}{default}{comma}\n"
        schema += ")"
        conn.close()
        return schema
    else:
        view_check_query = """
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.VIEWS 
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = ?
        """
        cursor.execute(view_check_query, (table_name,))
        view_exists = cursor.fetchone()[0] > 0
        if view_exists:
            view_def_query = f"""
            SELECT VIEW_DEFINITION
            FROM INFORMATION_SCHEMA.VIEWS
            WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}'
            """
            cursor.execute(view_def_query)
            view_def = cursor.fetchone()[0]
            schema = f"CREATE VIEW [dbo].[{table_name}] AS\n{view_def}"
            conn.close()
            return schema
        conn.close()
        raise HTTPException(status_code=404, detail=f"The {table_name} table/view does not exist")

# Updated function for Azure OpenAI API calls
def get_completion_from_azure_openai(prompt, temperature=0.5, max_tokens=1000):
    """Get completion from Azure OpenAI API with improved error handling and debugging"""
    try:
        # Debug information
        print(f"Azure OpenAI Request:")
        print(f"- Temperature: {temperature}")
        print(f"- Max tokens: {max_tokens}")
        print(f"- Prompt (first 100 chars): {prompt[:100]}...")
        
        if is_new_version:
            # For OpenAI SDK v1.x
            print("Using OpenAI SDK v1.x")
            
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            print(f"Response received (first 100 chars): {response_content[:100]}...")
            return response_content
        else:
            # For OpenAI SDK v0.x
            print("Using OpenAI SDK v0.x")
            
            response = openai.ChatCompletion.create(
                deployment_id=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            print(f"Response received (first 100 chars): {response_content[:100]}...")
            return response_content.strip()
            
    except Exception as e:
        error_message = f"Error in get_completion_from_azure_openai: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()  # Print the full stack trace
        
        # Add more specific error handling
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            error_details = "Authentication error: Please check your Azure OpenAI API key"
        elif "resource" in str(e).lower() or "endpoint" in str(e).lower():
            error_details = "Resource error: Please check your Azure OpenAI endpoint URL"
        elif "deployment" in str(e).lower() or "model" in str(e).lower():
            error_details = f"Deployment error: Please check if the deployment '{AZURE_OPENAI_DEPLOYMENT_NAME}' exists"
        elif "rate limit" in str(e).lower():
            error_details = "Rate limit exceeded: Please try again later"
        elif "timed out" in str(e).lower():
            error_details = "Request timed out: Please try again"
        else:
            error_details = str(e)
        
        raise HTTPException(status_code=500, detail=f"Azure OpenAI API error: {error_details}")

def get_nl2sql_response(question, table_name, schema, user_role=None):
    role_context = f"\nNote that this query is being made by a user with {user_role} role. " if user_role else ""
    if user_role and user_role != "Admin":
        if any(op in question.lower() for op in ["delete", "drop", "truncate", "update", "insert", "create"]):
            raise HTTPException(status_code=403, detail="You don't have permission to perform data modification operations")
    prompt = f"""Given the following SQL Server database schema:
{schema}

Convert this question into a SQL query to run against the {table_name} table/view:
{question}{role_context}

Return only the SQL query without any explanation or additional text.
"""
    sql_response = get_completion_from_azure_openai(prompt, temperature=0.1, max_tokens=500)
    
    if "```sql" in sql_response:
        sql_match = re.search(r'```sql(.+?)```', sql_response, re.DOTALL)
        if sql_match:
            sql_response = sql_match.group(1).strip()
    elif "```" in sql_response:
        sql_match = re.search(r'```(.+?)```', sql_response, re.DOTALL)
        if sql_match:
            sql_response = sql_match.group(1).strip()
    if "LIMIT" in sql_response:
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_response, re.IGNORECASE)
        if limit_match:
            limit_num = limit_match.group(1)
            sql_response = re.sub(r'LIMIT\s+\d+', '', sql_response, flags=re.IGNORECASE)
            sql_response = sql_response.replace("SELECT", f"SELECT TOP {limit_num}", 1)
    return sql_response

def execute_sql_query(sql_query):
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    cursor = conn.cursor()
    
    # Print the SQL query being executed
    print(f"Executing SQL query: {sql_query}")
    
    try:
        cursor.execute(sql_query)
        columns = [column[0] for column in cursor.description] if cursor.description else []
        results = []
        if columns:
            rows = cursor.fetchall()
            for row in rows:
                result_row = {}
                for i, value in enumerate(row):
                    if isinstance(value, (bytes, bytearray)):
                        value = "<binary data>"
                    elif hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    result_row[columns[i]] = value
                results.append(result_row)
        conn.close()
        
        # Print number of results
        print(f"Query returned {len(results)} results")
        
        return results
    except Exception as e:
        conn.close()
        error_message = f"SQL query execution error: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=400, detail=error_message)

def convert_results_to_natural_language(results, question, table_name):
    """Convert SQL results to natural language using Azure OpenAI"""
    if not results:
        return "I didn't find any data matching your query."
    
    # Serialize the results to JSON for the prompt
    results_json = json.dumps(results, indent=2)
    
    prompt = f"""Here are the results of a query against the {table_name} table:
```json
{results_json}
```

The original question was: "{question}"

Please convert these SQL query results into a natural language response that directly answers the question.
Make your response conversational and friendly. Focus on the key information and insights. 
Only mention specific numbers if they're significant to the answer.
DO NOT mention SQL, queries, or tables in your response.
"""
    
    natural_language = get_completion_from_azure_openai(prompt, temperature=0.5, max_tokens=1000)
    return natural_language

@app.post("/api/conversational-response")
async def get_conversational_response(request: ConversationalRequest):
    """Endpoint to get conversational responses from Azure OpenAI"""
    try:
        print(f"Received conversational request from user: {request.userEmail}")
        
        response = get_completion_from_azure_openai(
            request.prompt,
            temperature=0.7,  # Slightly higher temperature for more varied responses
            max_tokens=1000
        )
        
        return {"success": True, "message": response}
    except Exception as e:
        error_message = f"Error in conversational-response: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/api/classify-message")
async def classify_message(request: MessageClassificationRequest):
    """Endpoint to classify if a message is conversational or a database query"""
    try:
        print(f"Received classification request for message: {request.message[:50]}...")
        
        # Format a list of tables for context
        tables_list = ", ".join(request.accessibleTables[:10])  # Limit to first 10 for brevity
        additional_tables = f" and {len(request.accessibleTables) - 10} more" if len(request.accessibleTables) > 10 else ""
        
        # Create classification prompt
        prompt = f"""You are an AI assistant that helps classify user messages. 
Determine if the following message is a casual conversation or a database query.

User message: "{request.message}"

Available database tables: {tables_list}{additional_tables}

Classify the message as either:
- "conversational": general conversation, greetings, small talk, questions about you, etc.
- "database_query": any question that seems to be asking for information from a database

Only respond with one word: either "conversational" or "database_query"."""

        # Send to Azure OpenAI for classification
        response_text = get_completion_from_azure_openai(
            prompt,
            temperature=0.1,  # Low temperature for more consistent classification
            max_tokens=20     # Short response expected
        ).strip().lower()
        
        print(f"Classification result: {response_text}")
        
        # Clean up the response to ensure we get exactly what we need
        if "conversational" in response_text:
            message_type = "conversational"
        else:
            message_type = "database_query"
        
        return {"success": True, "type": message_type}
    except Exception as e:
        error_message = f"Error in classify-message: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "type": "database_query"}

@app.post("/api/login")
async def login(request: LoginRequest):
    try:
        print(f"Login attempt for email: {request.email}")
        roles = get_user_role(request.email)
        all_database_objects = get_all_database_objects()
        accessible_tables = []
        for role in roles:
            role_tables = define_table_access_by_role(role["name"], all_database_objects)
            accessible_tables.extend(role_tables)
        
        accessible_tables = list(set(accessible_tables))
        print(f"User {request.email} has access to {len(accessible_tables)} tables")
        
        return {
            "success": True,
            "email": request.email,
            "roles": roles,
            "accessibleTables": accessible_tables
        }
    except Exception as e:
        error_message = f"Login error: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/api/ask-llama")
async def ask_llama(request: AskLlamaRequest):
    try:
        print(f"Received ask-llama request from user: {request.userEmail}")
        print(f"Question: {request.question}")
        
        table_name = None
        format_type = request.format if hasattr(request, "format") else "text"  # Default to text
        
        # Check if format is specified in the question
        if "tabular format" in request.question.lower() or "table format" in request.question.lower():
            format_type = "table"
            print("Format detected in question: table")
        
        if request.question.startswith("[Table:"):
            match = re.match(r'\[Table: ([^\]]+)\] (.+)', request.question)
            if match:
                table_name = match.group(1)
                question = match.group(2)
                print(f"Table specified in question: {table_name}")
        else:
            question = request.question
        
        if table_name and table_name not in request.accessibleTables and "Admin" not in [role["name"] for role in request.userRoles]:
            print(f"Permission denied: User doesn't have access to table {table_name}")
            return {"success": False, "error": f"You don't have permission to query the {table_name} table"}
        
        if not table_name:
            print("No table specified in the question")
            return {"success": False, "error": "Please specify a table to query"}
        
        try:
            # Get the schema for the specified table
            schema = get_table_schema(table_name)
            print(f"Retrieved schema for table {table_name}")
            
            # Get the user's role
            user_role = request.userRoles[0]["name"] if request.userRoles else None
            print(f"User role: {user_role}")
            
            # Generate SQL query from natural language
            sql_query = get_nl2sql_response(question, table_name, schema, user_role)
            print(f"Generated SQL query: {sql_query}")
            
            if sql_query.startswith("Error:"):
                return {"success": False, "error": sql_query}
            
            # Execute the SQL query
            results = execute_sql_query(sql_query)
            
            if not results:
                print("Query returned no results")
                return {"success": True, "results": [], "message": "No results found for your query."}
            
            # For table format, return the raw results
            if format_type == "table":
                print(f"Returning results in table format ({len(results)} rows)")
                return {"success": True, "results": results, "format": "table"}
            
            # For text format, convert to natural language
            print("Converting results to natural language")
            natural_language_response = convert_results_to_natural_language(results, question, table_name)
            print(f"Natural language response (first 100 chars): {natural_language_response[:100]}...")
            return {"success": True, "message": natural_language_response, "format": "text"}
            
        except Exception as e:
            error_message = f"Error processing ask-llama request: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    except Exception as e:
        error_message = f"Unexpected error in ask-llama: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/api/tables")
async def get_tables():
    """Endpoint to get all available tables and views"""
    try:
        all_database_objects = get_all_database_objects()
        print(f"Retrieved {len(all_database_objects)} tables/views from database")
        return {"success": True, "tables": all_database_objects}
    except Exception as e:
        error_message = f"Error in get-tables: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/api/table-schema/{table_name}")
async def get_schema(table_name: str):
    """Endpoint to get the schema for a specific table or view"""
    try:
        print(f"Getting schema for table: {table_name}")
        schema = get_table_schema(table_name)
        return {"success": True, "schema": schema}
    except Exception as e:
        error_message = f"Error getting schema for {table_name}: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/api/health")
async def health_check():
    """Endpoint to check if the API is running and OpenAI connection works"""
    try:
        # Simple prompt to test OpenAI connection
        response = get_completion_from_azure_openai("Say hello", temperature=0.1, max_tokens=10)
        
        # Check database connection
        conn, error = establish_connection()
        db_status = "connected" if conn else f"error: {error}"
        if conn:
            conn.close()
        
        return {
            "status": "healthy",
            "openai_connection": "working",
            "openai_response": response,
            "database_connection": db_status,
            "api_version": "1.0.0"
        }
    except Exception as e:
        error_message = f"Health check failed: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return {
            "status": "unhealthy",
            "error": str(e),
            "api_version": "1.0.0"
        }

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)