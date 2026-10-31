# AI Chat Assistant (Vanilla Version)

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
You will also have to create a config file at `agent/config.json`.
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

## Notes
The app will automatically create a `config.json` and various `log-[session-id].txt`.
The `config.json` saves your chat settings across sessions.
The `log-[session-id].txt` logs all request/response payloads, and resets when a new chat session is created.
Please note that there will be residual logs and configs every time the server is restarted.
