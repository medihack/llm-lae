# LLM-LAE

- Install depdendencies with `uv sync`
- Create folder `env` and put data in there
- Copy `example.env` to `.env` and adjust values
- Run with `uv run analyze.py`
- To run ChatGPT
  - Add real `OPEN_AI_KEY` to `.env` file
- To run Ollama model
  - Make sure `ollama` is installed
  - Add model with more context `ollama create llama3.3-llm-lae -f Modelfile`
  - uncomment `OPEN_AI_BASE_URL` and set correct `OPEN_AI_MODEL` in `.env`
