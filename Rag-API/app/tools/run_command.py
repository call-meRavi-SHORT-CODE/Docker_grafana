import subprocess
from langchain.tools import tool

@tool
def run_command_tool(cmd: str) -> str:
    """
    Executes a shell command (e.g., a Docker CLI command) and returns stdout/stderr.
    Raises a RuntimeError on non-zero exit codes.
    """
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Execution failed:\n{proc.stderr}")
    return proc.stdout
