# LLM-LAE

## Preparation

- Install dependencies with `uv sync`
- Create folder `env` and put data in there
- Copy `example.env` to `.env` and adjust values accordingly.
- Make sure the Ollama is installed and the models to use are pulled.

## Usage

- Call `uv run extract --rules` for rule-based data extraction
- Call `uv run extract --llm model` for LLM data extraction using the provided model.
- Call `uv run extract -h` to see all options

## LLM support

`uv run extract --llm model` uses the OpenAI client or the Ollama client dependent of
the selected model (e.g. "gpt-4o" for OpenAI or "falcon3:70b" for Ollama). No further
configuration is needed (no Modelfile anymore).
