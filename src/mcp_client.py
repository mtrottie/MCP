import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from utils.helpers import validate_models_access
from dotenv import load_dotenv

PREFIX = """
Answer the following questions as best you can. You have access to the following tools:

{tool_names}
"""

FORMAT_INSTRUCTIONS = """
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: Select the tool by returning the name from the list of tools. Do not provide parameters here.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

SUFFIX = """
Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        validate_models_access(["anthropic.claude-3-sonnet-20240229-v1:0"])
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = ChatBedrockConverse(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=0,
            max_tokens=None,
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        tools = await load_mcp_tools(self.session)
        tool_strings = "\n".join(f"{tool.name}: {tool.description}" for tool in tools)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(PREFIX.format(tool_names=tool_strings)),
            HumanMessagePromptTemplate.from_template(FORMAT_INSTRUCTIONS),
            HumanMessagePromptTemplate.from_template(SUFFIX),
        ])
    
        tools = await load_mcp_tools(self.session)
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)        
        response = await agent_executor.ainvoke({"input": query}, include_run_info=True)
        return "\n".join(response["output"][0]["text"].split("\n"))

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())