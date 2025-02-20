import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import re

# Load environment variables
load_dotenv()

# Setting up the Streamlit app
st.set_page_config(page_title="Surya's Text to Math Problem Solver and Data Research Assistant")
st.title("Text to Math Problem Solver")

# Get Groq API Key from sidebar
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")
if not groq_api_key:
    st.info("Please type your GROQ API key to continue")
    st.stop()

# Initialize the language model
try:
    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize model: {e}")
    st.stop()

# Initialize the Chat Tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching Wikipedia to assist with math problems"
)

# Initialize the Math tool with enhanced explanation
try:
    math_chain = LLMMathChain.from_llm(llm=llm)
    
    # Define a custom math prompt to enforce numerical expressions
    math_prompt = """
    You are a mathematical assistant. For the given question, provide a numerical expression (no variables) and a detailed, point-wise explanation of how to solve it. If the question contains variables, assume reasonable numerical values (e.g., k=1) and state your assumption:
    Question: {question}
    Numerical Expression: <expression>
    Description:
    - Step 1: [First step]
    - Step 2: [Second step]
    - ... [Continue as needed]
    Final Result: <result>
    """
    math_prompt_template = PromptTemplate(input_variables=["question"], template=math_prompt)
    math_explain_chain = LLMChain(llm=llm, prompt=math_prompt_template)

    def calculate_with_explanation(question):
        try:
            # Get explanation from LLM
            explanation = math_explain_chain.run({"question": question})
            # Extract numerical expression
            expr_match = re.search(r'Numerical Expression: (.*?)\n', explanation)
            if not expr_match:
                return f"Question: {question}\nError: No valid numerical expression provided\nDescription:\n- Step 1: Failed to parse a numerical expression\nFinal Result: N/A"
            expr = expr_match.group(1).strip()
            # Check for variables and handle them
            if re.search(r'[a-zA-Z]', expr):
                assumption = "Assumption: Any variables (e.g., k) set to 1 unless specified."
                expr = re.sub(r'[a-zA-Z]', '1', expr)  # Replace variables with 1
                explanation = f"{explanation}\n{assumption}"
            result = math_chain.run(expr)
            return f"{explanation}\nFinal Result: {result}"
        except Exception as e:
            return f"Question: {question}\nNumerical Expression: {expr if 'expr' in locals() else 'N/A'}\nDescription:\n- Error: {e}\nFinal Result: N/A"

    calculator = Tool(
        name="Calculator",
        func=calculate_with_explanation,
        description="Solves math questions with step-by-step explanations."
    )
except Exception as e:
    st.error(f"Math tool initialization failed: {e}")
    st.stop()

# Define the reasoning prompt with step-by-step description
prompt = """
You are an agent tasked with solving the user's mathematical question. Provide a numerical expression (no variables) and a detailed, point-wise explanation. If variables are present, assume reasonable values (e.g., k=1) and note the assumption:
Question: {question}
Numerical Expression: <expression>
Description:
- Step 1: [First step]
- Step 2: [Second step]
- ... [Continue as needed]
Final Result: <result>
"""
prompt_template = PromptTemplate(input_variables=["question"], template=prompt)

# Combine tools into a chain
try:
    chain = LLMChain(llm=llm, prompt=prompt_template)
    reasoning_tool = Tool(
        name="Reasoning",
        func=chain.run,
        description="Answers math questions with step-by-step explanations."
    )
except Exception as e:
    st.error(f"Reasoning tool setup failed: {e}")
    st.stop()

# Initialize the agent
try:
    assistant_agent = initialize_agent(
        tools=[wikipedia_tool, calculator, reasoning_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )
except Exception as e:
    st.error(f"Agent initialization failed: {e}")
    st.stop()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math Chatbot who can solve your math questions with step-by-step explanations!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to generate the response
def generate_response(user_question):
    try:
        response = assistant_agent.invoke({'input': user_question})
        resp = response['output'] if isinstance(response, dict) and 'output' in response else str(response)
        if "Description:" not in resp:
            return calculate_with_explanation(user_question)  # Fallback for steps
        return resp
    except Exception as e:
        return f"Error: {e}"

# Interaction logic
question = st.text_area("Enter your Question:")

if st.button("Find my answer"):
    if question and question.strip():
        with st.spinner("Generating Response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = generate_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("### Response:")
            st.success(response)
    else:
        st.warning("Please enter a question")