from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, file_path):
        self.spp = SentencePieceProcessor(model_file=file_path)
        pass

    def encode(self,
               text: str,
               prepend_bos: bool,
               append_eos: bool
              ) -> list[int]:
        assert type(text) is str
        tokens = self.spp.encode(text)
        if prepend_bos:
            tokens = [self.bos_id] + tokens
        if append_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.spp.decode(tokens)

    @property
    def bos_id(self):
        return self.spp.bos_id()

    @property
    def eos_id(self):
        return self.spp.eos_id()

    @property
    def pad_id(self):
        return self.spp.pad_id()


if __name__ == '__main__':
    tokenizer = Tokenizer('/project/llama-2/tokenizer.model')

    text = 'Hello world!世界你好！'
    tokens = tokenizer.encode(text, False, False)
    print(tokens)
    tokens = tokens[:4] + [tokenizer.eos_id]
    print(tokens)
    print(tokenizer.decode(tokens))

