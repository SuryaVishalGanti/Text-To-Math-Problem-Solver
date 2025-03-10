Below is the revised version with bullet points instead of symbols:

---

**Surya's Text to Math Problem Solver & Research Assistant**

**Overview**  
This is an AI-powered Math Problem Solver & Research Assistant built using Streamlit and Gradio, leveraging LangChain and Groq's LLM. The application can:  
- Solve math problems with step-by-step explanations.  
- Extract and evaluate numerical expressions.  
- Search Wikipedia for additional context.  
- Perform reasoning-based calculations.  
- Accept text-based queries for mathematical problem-solving.  
- Run on both Streamlit (interactive UI) and Gradio (web-based interface).

---

**Features**  
- Step-by-Step Math Solutions – Detailed explanation for math problems.  
- Automated Numerical Expression Extraction – Parses and evaluates mathematical expressions.  
- Wikipedia Search Integration – Fetches additional knowledge for contextual understanding.  
- Works with Variables – Assumes values if needed and clearly states assumptions.  
- Dual Interface Support – Available in both Streamlit and Gradio.  
- Groq LLM Integration – Uses gemma2-9b-it for processing queries.

---

**Tech Stack**

- **Python:** Core programming language.  
- **Streamlit:** Interactive UI for user input and responses.  
- **Gradio:** Web-based interface for easy access.  
- **LangChain:** Manages LLM-based operations.  
- **Groq:** AI-powered inference for generating solutions.  
- **WikipediaAPIWrapper:** Fetches relevant information from Wikipedia.  
- **Regular Expressions (re):** Extracts numerical expressions from responses.  
- **dotenv:** Loads API keys securely.

---

**Functionality Breakdown**

1. **Model Initialization**  
   - Loads the Groq API key from the .env file or user input.  
   - Initializes Groq’s LLM (gemma2-9b-it) for processing questions.

2. **Tools & Features**  
   - **Math Problem Solver**  
     - Extracts numerical expressions from queries.  
     - Assumes default values for variables if unspecified.  
     - Provides step-by-step breakdown of the solution.
   - **Wikipedia Research Tool**  
     - Searches Wikipedia for additional mathematical context.  
     - Useful for concept explanations related to the problem.
   - **Reasoning & Calculation**  
     - Uses LangChain’s LLMChain for reasoning-based answers.  
     - Falls back to manual calculations if a detailed breakdown is missing.

---

**Error Handling**

- **Missing API Key:** Displays an error and asks the user to enter one.  
- **Invalid Inputs:** Warns the user if no valid question is provided.  
- **No Numerical Expression Found:** Returns an error message if parsing fails.  
- **Unrecognized Variables:** Assumes default values (e.g., k=1) and states the assumption.

---

This format should be easy to follow and discuss during your interview. Let me know if you need any more changes or additional details!
