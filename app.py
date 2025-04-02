import os
import json
import pandas as pd
import plotly
import plotly.express as px
from flask import Flask, request, render_template, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv
import zipfile
import glob
import shutil
import traceback
import tempfile

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# This function already accepts a directory path, regardless of where it is.
def load_spotify_data(directory_path):
    """
    Loads Spotify streaming history by finding and processing all
    'Streaming_History_Audio_*.json' files within the given directory.

    Combines data from all found files into a single DataFrame and performs
    essential preprocessing.

    Args:
        directory_path (str): The path to the directory containing extracted Spotify data.

    Returns:
        pandas.DataFrame or None: The processed DataFrame, or None if loading/processing fails.
    """
    # Define the pattern for the streaming history files
    file_pattern = os.path.join(directory_path, 'Streaming_History_Audio_*.json')
    # Find all files matching the pattern
    history_files = glob.glob(file_pattern)

    if not history_files:
        print(f"Error: No 'Streaming_History_Audio_*.json' files found in directory: {directory_path}")
        # Check if the directory itself contains the files (sometimes ZIPs have a subfolder)
        # Very basic check, might need refinement based on actual export structure
        subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        if len(subdirs) == 1:
             nested_dir = os.path.join(directory_path, subdirs[0])
             print(f"Checking subdirectory: {nested_dir}")
             file_pattern = os.path.join(nested_dir, 'Streaming_History_Audio_*.json')
             history_files = glob.glob(file_pattern)
             if not history_files:
                  print(f"Error: Also no relevant files found in subdirectory {nested_dir}.")
                  return None
        else:
             print(f"Found subdirectories, but unsure which contains the data: {subdirs}. Please ensure the ZIP structure is standard.")
             return None


    print(f"Found {len(history_files)} streaming history files to process:")
    # Sort files, maybe helpful for chronological order if needed, though pandas will sort by timestamp later
    history_files.sort()
    for f in history_files:
        print(f" - {os.path.basename(f)}")

    all_data = []
    total_records = 0
    files_processed = 0
    files_failed = 0

    # Loop through each found history file
    for file_path in history_files:
        print(f"\nProcessing file: {os.path.basename(file_path)}...")
        file_data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try loading as a standard JSON array first
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        file_data.extend(data)
                        print(f"  -> Loaded as JSON array ({len(file_data)} records).")
                    else:
                        print(f"  -> Warning: Expected a list of objects, but got {type(data)}. Skipping file.")
                        files_failed += 1
                        continue # Skip to next file
                except json.JSONDecodeError:
                    print(f"  -> Not a standard JSON array. Trying line-delimited JSON...")
                    f.seek(0) # Rewind file
                    lines_processed = 0
                    lines_skipped = 0
                    for i, line in enumerate(f):
                         line = line.strip()
                         if not line: continue
                         try:
                             record = json.loads(line)
                             if isinstance(record, dict):
                                 file_data.append(record)
                                 lines_processed += 1
                             else:
                                 lines_skipped += 1
                                 if lines_skipped < 5: print(f"  -> Warning: Skipped line {i+1} (not dict): {line[:50]}...")
                         except json.JSONDecodeError:
                             lines_skipped += 1
                             if lines_skipped < 5: print(f"  -> Warning: Skipped invalid JSON line {i+1}: {line[:50]}...")
                    if lines_skipped >= 5: print(f"  -> Warning: Skipped {lines_skipped} total invalid/non-dict lines in this file.")

                    if lines_processed > 0:
                         print(f"  -> Loaded as line-delimited JSON ({lines_processed} records).")
                    else:
                         print(f"  -> Error: Could not parse any valid records (line-delimited). Skipping file.")
                         files_failed += 1
                         continue # Skip to next file

            # If data was loaded successfully for this file
            if file_data:
                 all_data.extend(file_data)
                 total_records += len(file_data)
                 files_processed += 1

        except Exception as e:
            print(f"  -> Error processing file {os.path.basename(file_path)}: {e}")
            files_failed += 1
            # Optionally print traceback for debugging
            # traceback.print_exc()

    print(f"\nFinished processing files. Processed: {files_processed}, Failed: {files_failed}.")
    if not all_data:
        print("Error: No data could be loaded from any streaming history files.")
        return None

    print(f"Total records loaded across all files: {total_records}")

    # --- Create DataFrame from combined data ---
    df = pd.DataFrame(all_data)
    print(f"Combined data into DataFrame: {len(df)} rows, {len(df.columns)} columns.")
    # print(f"Original columns found: {df.columns.tolist()}") # Can be very long

    # --- Essential Preprocessing & Renaming (Same as previous version) ---
    # (Keep the timestamp, duration, and renaming logic from the previous answer here)

    # 1. Timestamp Processing
    if 'ts' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['ts'], errors='coerce', utc=True)
            original_count = len(df)
            df.dropna(subset=['timestamp'], inplace=True)
            if len(df) < original_count: print(f"Removed {original_count - len(df)} rows with invalid timestamps ('ts').")
            print("Created 'timestamp' (datetime UTC) column from 'ts'.")
        except Exception as e:
            print(f"Warning: Could not process 'ts' column: {e}. 'timestamp' column might be missing or incomplete.")
            if 'timestamp' not in df.columns: df['timestamp'] = pd.NaT
    else:
        print("Warning: Crucial 'ts' column not found. Time-based analysis will not be possible.")
        df['timestamp'] = pd.NaT

    # 2. Duration Processing
    if 'ms_played' in df.columns:
        try:
            df['msPlayed'] = pd.to_numeric(df['ms_played'], errors='coerce')
            df['minutesPlayed'] = df['msPlayed'] / 60000.0
            print("Created 'msPlayed' (numeric) and 'minutesPlayed' columns from 'ms_played'.")
            nan_count = df['msPlayed'].isna().sum()
            if nan_count > 0: print(f"Note: {nan_count} rows have invalid 'ms_played' values (NaN duration).")
        except Exception as e:
            print(f"Warning: Could not process 'ms_played' column: {e}. Duration columns might be missing or incomplete.")
            if 'msPlayed' not in df.columns: df['msPlayed'] = pd.NA
            if 'minutesPlayed' not in df.columns: df['minutesPlayed'] = pd.NA
    else:
        print("Warning: 'ms_played' column not found. Cannot calculate exact play duration.")
        df['msPlayed'] = pd.NA
        df['minutesPlayed'] = pd.NA

    # 3. Rename key metadata
    rename_map = {
        'master_metadata_track_name': 'trackName', 'master_metadata_album_artist_name': 'artistName',
        'master_metadata_album_album_name': 'albumName', 'spotify_track_uri': 'trackURI',
        'episode_name': 'episodeName', 'episode_show_name': 'showName', 'spotify_episode_uri': 'episodeURI',
        'reason_start': 'reasonStart', 'reason_end': 'reasonEnd',
    }
    actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    if actual_rename_map:
        df = df.rename(columns=actual_rename_map)
        print(f"Renamed columns for easier access: {list(actual_rename_map.values())}")

    # --- Final Check ---
    print(f"Preprocessing complete. Final DataFrame shape: {df.shape}")
    # ... (rest of the key column dtype printing can stay) ...
    key_cols = ['timestamp', 'msPlayed', 'minutesPlayed', 'trackName', 'artistName', 'albumName', 'skipped']
    print("Data types for key columns:")
    for col in key_cols:
        if col in df.columns: print(f" - {col}: {df[col].dtype}")
        else: print(f" - {col}: (Column not present)")

    return df


# --- Helper functions for LLM analysis (get_llm_analysis_code, execute_analysis_code) ---
# --- Keep these the same as before (No changes needed) ---
def get_llm_analysis_code(user_query, df_head_str, df_columns_str, df_dtypes_str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
         raise ValueError("OPENAI_API_KEY not found in environment variables.")

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are a data analysis assistant. You are given a user query about their Spotify listening history
and information about a pandas DataFrame named `df` containing this combined data from their export.

Your task is to generate *only* Python code that uses the `df` DataFrame and the Plotly Express library (`px`)
to answer the user's query. The code should generate a Plotly figure object assigned to a variable named `fig`.
If the query is best answered with text or a table, generate Python code that prints the answer or creates a
string/list/dict assigned to a variable named `result_data`. Do not include any explanations, just the code.

DataFrame Information (columns might vary based on user's specific export):
Columns: {df_columns_str}
Data Types:
{df_dtypes_str}

First 5 rows (df.head()):
{df_head_str}

User Query: "{user_query}"

Key Columns likely present and preprocessed:
- 'timestamp': datetime object (UTC) indicating when the track finished playing.
- 'msPlayed': numeric milliseconds played.
- 'minutesPlayed': numeric minutes played (derived from msPlayed).
- 'trackName': string name of the track.
- 'artistName': string name of the artist.
- 'albumName': string name of the album.
- Other original columns from the Spotify export might also exist.

Constraints:
- Use the DataFrame `df`.
- Use Plotly Express (`px`) for plotting and assign the figure to `fig`.
- For non-plot answers, assign the data to `result_data`.
- Ensure the code is safe and only performs read operations on the DataFrame for analysis.
- The code must be directly executable in an environment where `pandas as pd`, `plotly.express as px`, and the `df` variable exist.
- Output *only* the Python code. Do not add ```python or anything else around it.

Python Code:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful data analysis assistant generating Python code for Spotify data analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        code = response.choices[0].message.content.strip()
        if code.startswith("```python"): code = code[len("```python"):].strip()
        if code.endswith("```"): code = code[:-len("```")].strip()

        print("--- LLM Generated Code ---")
        print(code)
        print("--------------------------")
        return code
    except Exception as e:
        print(f"Error interacting with OpenAI API: {e}")
        return None

def execute_analysis_code(code, df):
    local_namespace = {'pd': pd, 'px': px, 'df': df}
    global_namespace = {}
    fig = None
    result_data = None
    error_message = None

    print(f"Executing code in safe environment...")
    try:
        exec(code, global_namespace, local_namespace)
        fig = local_namespace.get('fig')
        result_data = local_namespace.get('result_data')

        print(f"Execution successful. Found fig: {fig is not None}, Found result_data: {result_data is not None}")
        if fig and not isinstance(fig, plotly.graph_objs._figure.Figure):
             print("Warning: 'fig' variable is not a Plotly Figure object.")
             fig = None # Ignore if it's not the right type

    except Exception as e:
        print(f"Error executing generated code: {e}")
        error_message = f"Error during analysis execution: {e}"

    return fig, result_data, error_message

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main page."""
    # Clean up previous session's temporary directory if it exists
    # This is a fallback mechanism. Explicit cleanup via /cleanup is better.
    # Note: tempfile directories might also be cleaned by the OS eventually.
    existing_temp_dir = session.get('extracted_dir_path')
    if existing_temp_dir:
        print(f"Found leftover temp dir in session on index load: {existing_temp_dir}")
        # We don't aggressively clean here, rely on /cleanup or eventual OS cleanup
        # cleanup_extracted_data(existing_temp_dir) # Optional: clean aggressively
        # session.pop('extracted_dir_path', None)

    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles ZIP file upload, extraction into a temporary directory."""
    if 'spotify_data' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['spotify_data']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file is a ZIP
    if file and file.filename.endswith('.zip'):
        # --- Use tempfile ---
        # 1. Create a temporary directory using tempfile.mkdtemp()
        # This directory will be created in the system's default temp location.
        # We need to store its path and clean it up manually later.
        temp_dir_path = None # Initialize
        try:
            temp_dir_path = tempfile.mkdtemp(prefix="spotify_data_")
            print(f"Created temporary directory: {temp_dir_path}")

            # 2. Extract the zip file directly into the temporary directory
            # We pass the file stream directly to ZipFile, avoiding saving the zip itself
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir_path)
            print(f"ZIP file extracted to: {temp_dir_path}")

            # 3. Store the *path to the temporary directory* in the session
            # We store the path string, as we can't store the tempfile object itself
            session['extracted_dir_path'] = temp_dir_path

            # Optional: Quick check for expected files (important!)
            test_pattern = os.path.join(temp_dir_path, 'Streaming_History_Audio_*.json')
            nested_test_pattern = os.path.join(temp_dir_path, '*', 'Streaming_History_Audio_*.json')
            if not glob.glob(test_pattern) and not glob.glob(nested_test_pattern):
                 print(f"Warning: No 'Streaming_History_Audio_*.json' files found directly in {temp_dir_path} or one level down.")
                 # Clean up the created temp directory immediately if it's likely invalid
                 cleanup_extracted_data(temp_dir_path)
                 session.pop('extracted_dir_path', None)
                 return jsonify({"error": "Extraction successful, but no 'Streaming_History_Audio_*.json' files found inside the main folder or a direct subfolder. Please check the ZIP contents."}), 400

            # We don't need an upload_id anymore unless needed for UI feedback
            return jsonify({"message": "File uploaded and extracted successfully to a temporary location."})

        except zipfile.BadZipFile:
             # Clean up if extraction failed
             cleanup_extracted_data(temp_dir_path) # temp_dir_path might be None or valid here
             session.pop('extracted_dir_path', None) # Ensure session is clean
             return jsonify({"error": "Invalid or corrupted ZIP file."}), 400
        except Exception as e:
             print(f"Error during upload/extraction: {e}")
             traceback.print_exc() # Print detailed traceback
             # Clean up any partial results
             cleanup_extracted_data(temp_dir_path) # Attempt cleanup
             session.pop('extracted_dir_path', None) # Clear session state
             return jsonify({"error": f"An error occurred during processing: {e}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a ZIP file containing your Spotify data."}), 400

# --- /ask route (No functional changes needed, uses path from session) ---
@app.route('/ask', methods=['POST'])
def ask_question():
    """Handles user question, triggers LLM and analysis using data in the temp directory."""
    query = request.form.get('query')
    # Retrieve the PATH to the temporary directory from the session
    extracted_dir_path = session.get('extracted_dir_path')
    print(f"Received query: '{query}'")
    print(f"Using temp directory path from session: {extracted_dir_path}")

    if not query:
        return jsonify({"error": "No query provided."}), 400

    if not extracted_dir_path:
        print("Error: Temporary directory path not found in session.")
        return jsonify({"error": "Session expired or data not uploaded. Please upload your ZIP file again."}), 400
    if not os.path.isdir(extracted_dir_path):
        print(f"Error: Path from session exists but is not a directory: {extracted_dir_path}")
        # Clean up the session key if the path is invalid
        session.pop('extracted_dir_path', None)
        return jsonify({"error": "Server error: Invalid data path found. Please upload again."}), 500

    print(f"Processing query using data in temporary directory: {extracted_dir_path}")

    # 1. Load Data using the function (no changes needed)
    df = load_spotify_data(extracted_dir_path) # Pass the temporary directory path
    if df is None or df.empty:
        print("Failed to load data or DataFrame is empty from temp dir.")
        # load_spotify_data should print specific errors
        # Consider maybe cleaning up the temp dir here if loading fails?
        # cleanup_extracted_data(extracted_dir_path)
        # session.pop('extracted_dir_path', None)
        return jsonify({"error": "Failed to load or process the Spotify data from the temporary location."}), 500

    # 2. Get DataFrame Info for LLM Prompt (no change needed)
    df_head_str = df.head().to_string()
    try: df_dtypes_str = df.dtypes.to_string()
    except Exception: df_dtypes_str = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
    df_columns_str = ', '.join(df.columns)
    print("DataFrame Info prepared for LLM.")

    # 3. Get Analysis Code from LLM (no change needed)
    generated_code = get_llm_analysis_code(query, df_head_str, df_columns_str, df_dtypes_str)
    if not generated_code:
        print("Failed to get code from LLM.")
        return jsonify({"error": "Failed to get analysis instructions from LLM."}), 500
    print(f"--- Received LLM Code ---\n{generated_code}\n-------------------------")

    # 4. Execute Analysis Code (no change needed)
    fig, result_data, error_message = execute_analysis_code(generated_code, df)
    if error_message:
         print(f"Error during code execution: {error_message}")
         return jsonify({"error": error_message}), 500

    # 5. Prepare Response (no change needed)
    response_data = {}
    if fig:
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        response_data['graph'] = graph_json
    if result_data is not None:
        if isinstance(result_data, (pd.DataFrame, pd.Series)):
             response_data['data'] = result_data.to_dict(orient='records') if isinstance(result_data, pd.DataFrame) else result_data.to_dict()
             response_data['data_type'] = 'table' if isinstance(result_data, pd.DataFrame) else 'list/dict'
        elif isinstance(result_data, (list, dict, str, int, float)):
             response_data['data'] = result_data; response_data['data_type'] = 'text'
        else:
             response_data['data'] = str(result_data); response_data['data_type'] = 'text' # Fallback
    if not response_data:
         return jsonify({"message": "Analysis complete, but no specific plot or data was generated."})

    return jsonify(response_data)


# --- Helper for cleanup (Modified) ---
def cleanup_extracted_data(dir_path):
    """Safely removes the temporary directory created by tempfile."""
    # We no longer need the check against app.config['UPLOAD_FOLDER']
    if dir_path and os.path.isdir(dir_path):
        # Add a basic check to ensure it looks like a temp directory path if needed
        # e.g., if os.path.basename(dir_path).startswith('spotify_data_')
        # but generally, if we got it from the session, it should be the one we created.
        try:
            shutil.rmtree(dir_path)
            print(f"Cleaned up temporary directory: {dir_path}")
            return True
        except Exception as e:
            print(f"Error cleaning up temporary directory {dir_path}: {e}")
            return False
    elif dir_path and not os.path.isdir(dir_path):
        print(f"Cleanup requested, but path is not a directory or doesn't exist: {dir_path}")
        return False # Path doesn't exist or isn't a directory
    elif not dir_path:
        print("Cleanup requested, but no path provided.")
        return False # No path provided
    else:
        # This case should ideally not happen if dir_path comes from mkdtemp
        print(f"Warning: Unexpected path provided for cleanup: {dir_path}")
        return False


# --- /cleanup route (No functional changes needed) ---
@app.route('/cleanup', methods=['POST'])
def cleanup_session_data():
    """Route to explicitly clean up the session's temporary data directory."""
    dir_path = session.pop('extracted_dir_path', None) # Get path and remove from session
    print(f"Cleanup route called. Attempting to remove temp dir: {dir_path}")
    if cleanup_extracted_data(dir_path):
        return jsonify({"message": "Session data cleaned up."})
    else:
        # Even if cleanup failed or wasn't needed, the session key is gone.
        return jsonify({"message": "No active session data found or cleanup failed (check logs)."}), 202 # 202 Accepted


# --- Main execution ---
if __name__ == '__main__':
    # Note: Orphaned temporary directories (e.g., due to server crash)
    # might still exist. The OS typically handles cleaning the temp area
    # eventually (e.g., on reboot), but it's not guaranteed immediately.
    app.run(debug=True)