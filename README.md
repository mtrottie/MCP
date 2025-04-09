# Learning MCP

A basic MCP server using AWS bed rock.

## Getting started:

```
git clone git@github.com:mtrottie/MCP.git
```

Install python in the directory:
```
python -m vent .env
source .env/bin/activate
pip install -r requirements.txt
```

## Run the server

Set your AWS credentials via AWS env or put into a file named `.env`. Then run:

```
uv run src/mcp_client.py src/mcp_server.py
```
