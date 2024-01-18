from typing import List
from app.models.text_generation_lm import TextGenerationLM
import os
import subprocess
from threading import Thread

from app.streamer import TextStreamer


MODEL_NAME = "cli_streamer_echo"


# A class to treat a command line command as a stream
class CliLM(TextGenerationLM):
    def __init__(self, model_name: str):
        command = model_name[len("cli_streamer_") :]
        self.check_command = command
        self.run_command = command

        super().__init__(MODEL_NAME, has_tokenizer=False, load_from_s3=False)

    @classmethod
    def get_model_names(cls) -> List[str]:
        return [MODEL_NAME]

    def setup(self) -> None:
        subprocess.check_call(
            f"which {self.check_command}", shell=True, stdout=subprocess.DEVNULL
        )

    async def answer(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        verbose: bool = False,
        stream: bool = False,
    ) -> str:
        def stream_response(streamer: TextStreamer = None):
            full_text = ""
            process = subprocess.Popen(
                [self.run_command, f'"{prompt}"'],
                stdout=subprocess.PIPE,
            )

            def read_next_word(streamer):
                line_bytes = process.stdout.read(5)
                line = line_bytes.decode()
                if streamer is not None:
                    streamer.put(line)
                return line

            while process.poll() is None:
                full_text += read_next_word(streamer)

            full_text += read_next_word(streamer)

            if streamer is not None:
                streamer.end()
            return full_text

        if stream:
            streamer = TextStreamer()
            t = Thread(target=stream_response, kwargs=dict(streamer=streamer))
            t.start()

            return streamer
        else:
            return stream_response()
