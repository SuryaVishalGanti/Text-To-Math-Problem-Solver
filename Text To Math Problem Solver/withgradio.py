import gradio as gr
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
import re
import os

# Load environment variables
load_dotenv()

# Initialize the language model
def initialize_model(groq_api_key):
    try:
        llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
        return llm
    except Exception as e:
        raise Exception(f"Failed to initialize model: {e}")

# Define calculate_with_explanation globally
def calculate_with_explanation(question, llm):
    try:
        math_chain = LLMMathChain.from_llm(llm=llm)
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
        
        explanation = math_explain_chain.run({"question": question})
        expr_match = re.search(r'Numerical Expression: (.*?)\n', explanation)
        if not expr_match:
            return f"Question: {question}\nError: No valid numerical expression provided\nDescription:\n- Step 1: Failed to parse a numerical expression\nFinal Result: N/A"
        expr = expr_match.group(1).strip()
        if re.search(r'[a-zA-Z]', expr):
            assumption = "Assumption: Any variables (e.g., k) set to 1 unless specified."
            expr = re.sub(r'[a-zA-Z]', '1', expr)
            explanation = f"{explanation}\n{assumption}"
        result = math_chain.run(expr)
        return f"{explanation}\nFinal Result: {result}"
    except Exception as e:
        return f"Question: {question}\nNumerical Expression: {expr if 'expr' in locals() else 'N/A'}\nDescription:\n- Error: {e}\nFinal Result: N/A"

# Initialize the Chat Tools
def initialize_tools(llm):
    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = Tool(
        name="Wikipedia",
        func=wikipedia_wrapper.run,
        description="A tool for searching Wikipedia to assist with math problems"
    )

    # Math tool using the global calculate_with_explanation
    calculator = Tool(
        name="Calculator",
        func=lambda q: calculate_with_explanation(q, llm),  # Pass llm to the function
        description="Solves math questions with step-by-step explanations."
    )

    # Define the reasoning prompt
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

    try:
        chain = LLMChain(llm=llm, prompt=prompt_template)
        reasoning_tool = Tool(
            name="Reasoning",
            func=chain.run,
            description="Answers math questions with step-by-step explanations."
        )
    except Exception as e:
        raise Exception(f"Reasoning tool setup failed: {e}")

    return [wikipedia_tool, calculator, reasoning_tool]

# Function to generate the response
def generate_response(user_question, groq_api_key):
    try:
        llm = initialize_model(groq_api_key)
        tools = initialize_tools(llm)
        
        assistant_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )
        
        response = assistant_agent.invoke({'input': user_question})
        resp = response['output'] if isinstance(response, dict) and 'output' in response else str(response)
        if "Description:" not in resp:
            return calculate_with_explanation(user_question, llm)  # Use global function with llm
        return resp
    except Exception as e:
        return f"Error: {e}"

# Gradio Interface
def gradio_interface(question, api_key):
    if not api_key:
        return "Please provide a GROQ API key."
    if not question or not question.strip():
        return "Please enter a question."
    return generate_response(question, api_key)

# Create Gradio app
with gr.Blocks(title="Surya's Text to Math Problem Solver") as demo:
    gr.Markdown("# Text to Math Problem Solver")
    gr.Markdown("Enter your math question and GROQ API key to get a step-by-step solution!")
    
    api_key_input = gr.Textbox(label="GROQ API Key", type="password", placeholder="Enter your GROQ API key here")
    question_input = gr.Textbox(label="Enter your Question", placeholder="e.g., What is 5 + 3 * 2?")
    output = gr.Textbox(label="Response", interactive=False)
    
    submit_button = gr.Button("Find my answer")
    submit_button.click(
        fn=gradio_interface,
        inputs=[question_input, api_key_input],
        outputs=output
    )

# Launch the app
demo.launch(share=True)