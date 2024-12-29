"""This script is used to setup models with more context in the Ollama server.

This is necessary as currently it is not possible to provide the context length by
using the OpenAI client.
This can be fixed with https://github.com/ollama/issues/7063
"""

import ollama

falcon_model = """
FROM falcon3:10b

PARAMETER temperature 0
PARAMETER num_ctx 8192
"""

llama_model = """
FROM llama3.3:70b

PARAMETER temperature 0
PARAMETER num_ctx 8192
"""

qwen_model = """
FROM qwen2.5:72b

PARAMETER temperature 0
PARAMETER num_ctx 8192
"""

model_files = {
    "falcon3-lae:10b": falcon_model,
    "llama3.3-lae:70b": llama_model,
    "qwen2.5-lae:72b": qwen_model,
}


def main():
    client = ollama.Client(host="http://localhost:11434")
    for model_name, model_file in model_files.items():
        client.create(model=model_name, modelfile=model_file)


if __name__ == "__main__":
    main()
