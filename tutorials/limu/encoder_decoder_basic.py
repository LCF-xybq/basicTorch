from torch import nn

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, x, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_outputs = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_x, dec_state)