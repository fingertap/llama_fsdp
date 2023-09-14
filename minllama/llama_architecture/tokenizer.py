from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self,
                 file_path: str,
                 prepend_bos: bool = True,
                 append_eos: bool = True):
        self.spp = SentencePieceProcessor(model_file=file_path)

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
