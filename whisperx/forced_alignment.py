import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dtaidistance import dtw

class ForcedAlignment:
    def __init__(self, model_name, device):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.device = device

    def align(self, audio, transcription):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_phonemes = self.processor.batch_decode(predicted_ids)

        # Convert transcription to phonemes (this is a simplified approach)
        transcription_phonemes = self._text_to_phonemes(transcription)

        # Perform DTW alignment
        alignment = dtw.dtw(predicted_phonemes[0], transcription_phonemes)

        return self._process_alignment(alignment, len(audio) / 16000)

    def _text_to_phonemes(self, text):
        # This is a placeholder. In a real implementation, you'd use a 
        # text-to-phoneme converter appropriate for the language.
        return list(text.lower().replace(" ", ""))

    def _process_alignment(self, alignment, audio_duration):
        # Convert DTW alignment to time-based alignment
        time_step = audio_duration / len(alignment[0])
        return [(i * time_step, j * time_step) for i, j in zip(*alignment)]