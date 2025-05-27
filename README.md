# Legal documents advizor bot

This repository contains implementation of the bot that can do tasks:

- Ingest specific legal document and store it to database
- Answer questions related to previously ingested documents 


### Technology stack

- Python as programming language
- ChromaDb as vector database for storing embeddings. It is open-source and easy to use. It has integrations with llm frameworks.
- OpenAI as cloud platform providing relatively cheap, easy to use LLMs for various tasks
- Aiogram is simple and feature-rich telegram bot library

### How to deploy bot (locally)

1. Install requirements into your python environment.

```
pip install -r requirements.txt
```
2. Add required env variables to .env file into the root of the project.

 - OPENAI_API_KEY
 - TG_TOKEN
 - EMBEDDING_MODEL

3. Run the bot.

```
python app.py
```

### How to use bot in telegram

* `/start` - this command starts communication with the bot
* `/ingest` - this command is used when you upload new legal file for ingestion.

    1. Send a file for ingestion.

    2. Reply to that message and send: `/ingest` command.

* `/ask` - this command used to send questions to database and llm. Usage: `/ask <question>`


### Further improvements

* Deploy backend to the public cloud.
* Distribute ingestion requests for faster processing.
* Improve functionality of the boot building complex AI agents using frameworks like LangChain and LangGraph.
* Experiment with different LLM models.