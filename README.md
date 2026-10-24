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
  * Initial commits by Qwen Coder

## How to Run
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

## Notes
The app will automatically create a `config.json` and a `log.txt`.
The `config.json` saves your chat settings across sessions.
The `log.txt` logs all request/response payloads, and resets when a new chat session is created.
