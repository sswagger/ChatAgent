# AI Chat Assistant (MCP)

## Description
This is an AI chat agent that is compatible with openAI models.

* Language
  * English
  * Python
  * HTML
  * Javascript
  * CSS
* Type of Application
  * Web
* Author
  * Sam Swagger
  * Tim Swagger
  * Initial commits by Qwen Coder

## How to Run
### Docker
First create an `.env` file at `.docker\agent\`.
The `.env` file should look like the following:
```dotenv
LLM_API_URL="http://PATH.TO.YOUR/MODEL"
LLM_API_KEY="YOUR-LLM-API-KEY"
LLM_MODEL="NAME-OF-MODEL"
```

Then install [Docker](https://www.docker.com/products/docker-desktop/). Then run the following command at the project root to build the container:

```bash
docker compose -f ./docker-compose.yml up
```
This will create a docker container at `localhost:8080` that you can open in any web browser.

### Configs
You will also have to create two config files at `agent/config.json` and `agent/mcp_config.json`.
The contents should be as follows:

#### config.json
```json
{
  "default": {
    "systemPrompt": "Your default system prompt",
    "bufferSize": 10,
    "temperature": 0.0,
    "port": 8080
  }
}
```

#### mcp_config.json
```json
{
  "default": {
    "enabled": true,
    "url": "http://restaurant-mcp:8000/mcp",
    "connectionStatus": "disconnected",
    "tools": [
      {
        "name": "get_capabilities",
        "description": "Get a list of all available tools from the MCP server. Use this when someone asks what tools are available."
      },
      {
        "name": "get_time",
        "description": "Gets the current datetime"
      },
      {
        "name": "get_config",
        "description": "Gets the settings for the chat agent"
      },
      {
        "name": "get_timezone",
        "description": "Gets the servers timezone"
      },
      {
        "name": "log_message",
        "description": "Adds a message to the log file",
        "inputSchema": {
          "additionalProperties": false,
          "properties": {
            "sql": {
              "type": "text",
              "description": "(str) The text to log"
            }
          },
          "required": [
            "text"
          ],
          "type": "object"
        }
      }
    ],
    "lastError": null,
    "systemPrompt": "IMPORTANT: This application has access to an MCP (Model Context Protocol) server that provides tools. When asked what the MCP server can do, or what tools are available, you MUST call the get_capabilities function to retrieve the list of available tools. Do NOT just make up a response. The get_capabilities function will return a list of all available tools with their descriptions.\n"
  }
}
```

## Notes
The mcp tool calling is compatible with [fastmcp](https://gofastmcp.com/getting-started/welcome).
I have not tested it with other mcp tools (or servers).
I created a fastmcp server for database access that can be found on GitHub [here](https://github.com/sswagger/Restaurant-Stats).

The app will automatically create a `config.json` and various `log-[session-id].txt`.
The `config.json` saves your chat settings across sessions; the `mcp_config.json` does similar, only with mcp configurations.
The `log-[session-id].txt` logs all request/response payloads, and resets when a new chat session is created.
Please note that there will be residual logs and configs every time the server is restarted.