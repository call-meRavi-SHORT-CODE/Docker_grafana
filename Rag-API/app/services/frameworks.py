# app/services/framework_factory.py
import subprocess
from typing import Any
from app.tools.run_command import run_command_tool
from Prompt.prompts import system_prompt
from langgraph.prebuilt import create_react_agent

from llama_index.core.agent.workflow import FunctionAgent
import dspy
from dspy import InputField, OutputField, Signature




def get_agent(framework_name: str, llm, rag_chain):
    """
    Returns a ReACT‐style agent configured for the chosen framework.
    Currently supports "langgraph". AutoGen is a placeholder.
    """

    from typing import Dict
    from langchain.tools import tool
    from app.tools.doc_qa import doc_qa_tool as _raw_doc_qa_tool

    @tool
    def _doc_qa(query: str) -> str:
        """
        Retrieve an answer from the indexed documentation using RAG.
        """
        # Here, `rag_chain` is closed over from the outer scope.
        result: Dict[str, Any] = rag_chain.invoke({"input": query})
        return result.get("answer", "No answer found.")

    tools = [
        _doc_qa,            # Now a named function with its own docstring
        run_command_tool    # Already has a docstring in app/tools/run_command.py
    ]

    if framework_name == "langgraph":
        # Pass your actual system_prompt into create_react_agent
        return create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    elif framework_name == "autogen":
        # Placeholder for an AutoGen‐based agent
        raise       ("AutoGen framework not implemented yet")


    elif framework_name == "dspy":

        

        # 2. Define your tools (doc_qa and run_command must be implemented separately)
        def dspy_doc_qa(query: str) -> str:
            """
            Retrieve an answer from the indexed documentation using RAG.
            """
            # Here, `rag_chain` is closed over from the outer scope.
            result: Dict[str, Any] = rag_chain.invoke({"input": query})
            return result.get("answer", "No answer found.")
            
            
        def dspy_run_command(cmd: str) -> str:
            """
            Executes a shell command (e.g., a Docker CLI command) and returns stdout/stderr.
            Raises a RuntimeError on non-zero exit codes.
            """
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Execution failed:\n{proc.stderr}")
            return proc.stdout
        

        tools = [dspy_doc_qa, dspy_run_command]

        # 3. Build the Signature with your system_prompt in instructions
        sig = Signature(
            {"question": InputField()},
            instructions=system_prompt
        ).append("answer", OutputField(), type_=str)

        
        dspy_react = dspy.ReAct(signature=sig, tools=tools)
        return dspy_react
    
    elif framework_name == "llamaindex":


        #tool1
        def llamaindex_doc_qa(query: str) -> str:
            """
            Retrieve an answer from the indexed documentation using RAG.
            """
            # Here, `rag_chain` is closed over from the outer scope.
            result: Dict[str, Any] = rag_chain.invoke({"input": query})
            return result.get("answer", "No answer found.")
        
        def llamaindex_run_command_tool(cmd: str) -> str:
            """
            Executes a shell command (e.g., a Docker CLI command) and returns stdout/stderr.
            Raises a RuntimeError on non-zero exit codes.
            """
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Execution failed:\n{proc.stderr}")
            return proc.stdout
        

        
        
        
        return  FunctionAgent(
        tools=[llamaindex_doc_qa,llamaindex_run_command_tool],
        llm=llm,
        system_prompt=system_prompt,
    )

      

    else:
        raise ValueError(f"Unsupported framework: {framework_name}")