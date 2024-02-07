# Radiant Model Runtime

### _Run any LLM in any environment. Supports CPU and GPU model runtimes._

# Usage

To run the server run it either through `poetry`

```bash
poetry run start
```

or with `uvicorn`

```bash
poetry run uvicorn app.app:app --host 0.0.0.0 --port=8000 --reload
```

## API

The API is intentionally super simple. It support text generation, embedding, json extraction and fine tuning. Besides, it supports loading different model architectures. Because we use FastAPI and OpenAPI can found under `/docs`.

### Text generation

Parameters
```json
{
  "prompt": "string", # The prompt to run 
  "token_count": 100, # (Optional) Maximum token count
  "temperature": 0, # (Optional)
  "verbose": false, # (Optional) Log additional debug information
  "stream": false # (Optional)
}
```

Example
```bash
curl -XPOST http://localhost:8000/api/text_generation -H 'content-type: application/json' -d '{ "prompt": "123"}'
```


### Embedding

Parameters
```json
{
  "prompts": [
    "string" # A list of text strings to embedd.
  ]
}
```

Example
```bash
curl -XPOST http://localhost:8000/api/embedding -H 'content-type: application/json' -d '{ "prompts": ["123"]}'
```

### Fine tuning

Parameters
```bash
{
  "examples": [  # Example text that the model should be trained on
    "string"
  ],
  "steps": 100, # Training steps
  "base_model": "string", # LLM base model
  "name": "string" # Name of the new finetuned model
}
```

Example
```bash
curl -XPOST http://localhost:8000/api/finetunin/sft -H 'content-type: application/json' -d '{ "examples": ["123"], "steps": 10, "base_model": "llama2", "name": "finetuned_llama2"}'
```


# Features

- [x] Support loading models in GPU memory and offloading them when switching models
- [x] Supports loading and caching of models from S3
- [x] Supports the most popular open source models and runtime like llama2, Mistral, vLLM + Llama, Ollama
- [x] Supports SFT through a simple API and storing the adapter in S3

# Roadmap

- [ ] Add generic Huggingface transformer interface
- [ ] Add more finetuning strategies
- [ ] Support Azure
- [ ] Support GCP 

# Installation

```bash
# Make sure you have poetry and the respective libraries installed

poetry install
pip3 install install flash-attn==2.3.1.post1 --no-build-isolation
pip3 install "transformers[torch]"
```

## Questions?

Create an issue or discussion in this repository.

Or, reach out to our team! [@jakob_frick](https://twitter.com/frick_jakob/), [@__anjor](https://twitter.com/__anjor), [@maxnajork](https://twitter.com/maxnajork) on X or [team@radiantai.com](mailto:team@radiantai.com).

## Contributing Guidelines

Thank you for your interest in contributing to our project! Before you begin writing code, it would be helpful if you read these [contributing guidelines](CONTRIBUTING.md). Following them will make the contribution process easier and more efficient for everyone involved.

