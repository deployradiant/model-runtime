from typing import List
from app.models.text_generation_lm import TextGenerationLM
import requests
import subprocess
from threading import Thread
import json

from app.streamer import TextStreamer


AVAILABLE_MODELS = ["llama2", "zephyr"]
MODEL_NAME_STUB = "ollama-wrapper-"


class Ollama(TextGenerationLM):
    def __init__(self, model_name: str):
        if not model_name.startswith(MODEL_NAME_STUB):
            raise ValueError(f"Model name must start with {MODEL_NAME_STUB}")
        self.ollama_model = model_name[len(MODEL_NAME_STUB) :]

        super().__init__(model_name=model_name, has_tokenizer=False, load_from_s3=False)

    @classmethod
    def get_model_names(cls) -> List[str]:
        return [MODEL_NAME_STUB + model for model in AVAILABLE_MODELS]

    def setup(self) -> None:
        subprocess.check_call(f"ollama pull {self.ollama_model}", shell=True)
        self.proc = subprocess.Popen(
            "ollama serve",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def reset(self):
        self.proc.terminate()
        super().reset()

    async def answer(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        verbose: bool = False,
        stream: bool = False,
    ) -> str:
        def stream_ollama_response(streamer: TextStreamer = None):
            full_text = ""
            session = requests.Session()

            with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "prompt": prompt,
                    "model": self.ollama_model,
                    "temperature": temperature,
                },
                stream=stream,
            ) as response:
                if response.status_code != 200:
                    return ""

                for line in response.iter_lines():
                    parsed_response = parse_ollama_response_line(line)
                    if parsed_response is not None:
                        if streamer is not None:
                            streamer.put(parsed_response)
                        full_text += parsed_response

            if streamer is not None:
                streamer.end()
            session.close()
            return full_text

        if stream:
            streamer = TextStreamer()
            t = Thread(target=stream_ollama_response, kwargs=dict(streamer=streamer))
            t.start()

            return streamer
        else:
            return stream_ollama_response()


def parse_ollama_response_line(line):
    raw_json = line.decode("utf-8")
    chunks = raw_json.split("\n")
    for chunk in chunks:
        if chunk.strip() == "":
            continue
        parsed = json.loads(chunk)
        if "response" in parsed:
            return parsed["response"]
