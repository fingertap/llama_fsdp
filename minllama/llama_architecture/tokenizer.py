from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self,
                 file_path: str,
                 prepend_bos: bool = True,
                 append_eos: bool = True):
        self.spp = SentencePieceProcessor(model_file=file_path)
        self.apped_eos = append_eos
        self.prepend_bos = prepend_bos

    def encode(self,
               text: str,
               prepend_bos: bool = None,
               append_eos: bool = None
              ) -> list[int]:
        assert type(text) is str
        tokens = self.spp.encode(text)

        # Postprocess
        prepend_bos = self.prepend_bos is prepend_bos is None
        append_eos = self.append_eos is append_eos is None
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
