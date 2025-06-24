import re
from langchain_core.output_parsers import StrOutputParser


class DockerCommandParser(StrOutputParser):
    def parse(self, text: str) -> str:
        # If the response does not contain any newline characters, return it as-is
        if '\n' not in text:
            return text.strip()

        # Extract content between triple backticks
        code_blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        # Fallback: extract lines that start with 'docker'
        lines = text.strip().splitlines()
        command_lines = [line.strip() for line in lines if line.strip().startswith("docker")]
        return "\n".join(command_lines)


def get_parser():
    return DockerCommandParser()