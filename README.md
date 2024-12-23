# LLM-LAE

- Install dependencies with `uv sync`
- Create folder `env` and put data in there
- Copy `example.env` to `.env` and adjust values
- Run with `uv run workflow`
  - Limit reports to e.g. 3 `uv run workflow -l 3`
  - Limit to specific study IDs `uv run workflow -s CBS0003 -sCBS0099`
- To run ChatGPT
  - Add real `OPEN_AI_KEY` to `.env` file
- To run Ollama model
  - Make sure `ollama` is installed
  - Add model with more context `ollama create llama3.3-llm-lae -f Modelfile`
  - uncomment `OPEN_AI_BASE_URL` and set correct `OPEN_AI_MODEL` in `.env`
