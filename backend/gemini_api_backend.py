import os
import json
import datetime
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import google.generativeai as genai

# --- Pydantic Models ---

class AssignmentItem(BaseModel):
    name: str
    due_date: Optional[str] = None
    due_time: Optional[str] = None

class Assignments(BaseModel):
    assignments: List[AssignmentItem]

class ExtractRequest(BaseModel):
    text: str

# --- FastAPI App Setup ---

app = FastAPI(title="Gemini Assignment Extractor API")

# Add CORS middleware to allow your frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3000/deadline_tracker++",
        "https://shriyabi.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini API Configuration ---

# Load API Key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_KEY")
if not api_key:
    raise ValueError("GEMINI_KEY not found in .env file.")
genai.configure(api_key=api_key)

MODEL_NAME="gemini-2.5-flash-lite"
@app.post("/extract-assignments", response_model=Assignments)
def extract_assignments(req: ExtractRequest):
    """
    Receives text, extracts assignment details using Gemini, 
    and returns them as a structured JSON object.
    """
    model = genai.GenerativeModel(MODEL_NAME)

    # The prompt is updated with specific rules and examples (few-shot learning).
    prompt = f"""
        You are an intelligent assistant that extracts assignment information from text.
        Your task is to identify all assignments, quizzes, or deadlines and return them as a valid JSON array of objects.

        **Extraction Rules:**
        1.  'name': The full title of the assignment. The name is often the text on the first line of a section, before any due dates or point values. Do not use generic phrases like "Assignment Instructions" as the name.
        2.  'due_date': The due date in 'YYYY-MM-DD' format. Assume the current year is {datetime.date.today().year}.
        3.  'due_time': The due time in 24-hour 'HH:MM' format. If no time is specified, use null.
        4.  If an assignment is mentioned but has no clear due date (e.g., "Not available yet"), do not include it.
        5.  Ensure the output is ONLY the JSON array, with no extra text or markdown formatting.

        **Here are some examples:**

        **Example 1 Input Text:**
        ---
        Assignment 1: Project Proposal
        Due Oct 15 by 11:59pm
        100 pts
        
        Quiz 3: Chapter 5
        Available from Oct 10
        Due Oct 17
        ---
        **Example 1 Desired JSON Output:**
        ```json
        [
          {{
            "name": "Assignment 1: Project Proposal",
            "due_date": "{datetime.date.today().year}-10-15",
            "due_time": "23:59"
          }},
          {{
            "name": "Quiz 3: Chapter 5",
            "due_date": "{datetime.date.today().year}-10-17",
            "due_time": null
          }}
        ]
        ```

        **Now, analyze the real text provided below and generate the JSON output based on the rules and examples.**

        **Real Input Text:**
        ---
        {req.text}
        ---
    """
    
    print("Sending text to Gemini for extraction...")
    assignments_list = []
    try:
        response = model.generate_content(prompt)
        
        # It's good practice to print the raw response for debugging
        print("--- Raw Gemini Response ---")
        print(response.text)
        print("---------------------------")
        
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        assignments_list = json.loads(cleaned_response)
        print(f"Successfully extracted {len(assignments_list)} assignment(s).")

    except (json.JSONDecodeError, Exception) as e:
        print(f"Error processing Gemini response: {e}")
        assignments_list = []

    # This is the logic to duplicate events if a time is specified
    processed_assignments = []
    for assignment in assignments_list:
        due_time = assignment.get('due_time')
        if due_time:
            all_day_version = {
                "name": f"{assignment.get('name', 'Untitled')} ({due_time})",
                "due_date": assignment.get('due_date'),
                "due_time": None
            }
            processed_assignments.append(all_day_version)
            processed_assignments.append(assignment)
        else:
            processed_assignments.append(assignment)
            
    print(f"Returning {len(processed_assignments)} processed assignment(s).")
    return {"assignments": processed_assignments}

# To run the server: uvicorn gemini_api_backend:app --reload