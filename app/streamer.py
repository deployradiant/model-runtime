from queue import Queue


# Inspired by TextIteratorStreamer from HF Transformers
# from transformer import TextIteratorStreamer
class TextStreamer:
    def __init__(self):
        self.text_queue = Queue()
        self.timeout = None
        self.stop_signal = None

    def end(self):
        self.text_queue.put(self.stop_signal)

    def put(self, text):
        self.text_queue.put(text)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
