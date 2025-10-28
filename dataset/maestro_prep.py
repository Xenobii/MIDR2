import os
import json
import h5py
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from preprocessors.prep_wav import WavPreprocessor
from preprocessors.prep_midr import MidrPreprocessor
from preprocessors.prep_midi import MidiPreprocessor


class MaestroPreprocessor:
    def __init__(self, dir_maestro, prep_wav, prep_midi, prep_midr):
        self.maestro_dir = dir_maestro
        self.df          = pd.read_csv(os.path.join(self.maestro_dir, "maestro-v3.0.0.csv"))
        self.f_out       = "dataset/processed_dataset.h5"
        self.prep_midr   = prep_midr
        self.prep_wav    = prep_wav
        self.prep_midi   = prep_midi

    def process_all(self):
        with h5py.File(self.f_out, "w") as h5:
            print(f"Processing maestro v3... \n")
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
                midi_path = os.path.join(self.maestro_dir, row["midi_filename"])
                wav_path  = os.path.join(self.maestro_dir, row["audio_filename"])

                spec_chunks = self.prep_wav(wav_path)
                midi_chunks = self.prep_midi(midi_path)
                midr_chunks = self.prep_midr(midi_path)

                nchunks = min(
                    len(spec_chunks),
                    len(midi_chunks["mpe"]),
                    len(midr_chunks['spiral_cc'])
                )
                # wav will always be the longer one
                spec_chunks = spec_chunks[:nchunks]
                for key in midi_chunks:
                    midi_chunks[key] = midi_chunks[key][:nchunks]
                for key in midr_chunks:
                    midr_chunks[key] = midr_chunks[key][:nchunks]

                assert len(spec_chunks) == len(midr_chunks['spiral_cc']), \
                f"Incompatible chunk length: Spec: {len(spec_chunks)}, Midr: {len(midr_chunks['spiral_cc'])}"
                assert len(spec_chunks) == len(midi_chunks['mpe']), \
                f"Incompatible chunk length: Spec: {len(spec_chunks)}, Midr: {len(midi_chunks['mpe'])}"

                group = h5.create_group(f"{idx:07d}")
                group.create_dataset("spec", data=spec_chunks, compression="lzf")

                group.create_dataset("mpe", data=midi_chunks["mpe"], compression="lzf")
                group.create_dataset("onset", data=midi_chunks["onset"], compression="lzf")
                group.create_dataset("offset", data=midi_chunks["offset"], compression="lzf")
                group.create_dataset("velocity", data=midi_chunks["velocity"], compression="lzf")

                group.create_dataset("spiral_cd", data=midr_chunks["spiral_cd"], compression="lzf")
                group.create_dataset("spiral_cc", data=midr_chunks["spiral_cc"], compression="lzf")

                group.create_dataset("note_mask", data=midr_chunks["note_mask"], compression="lzf")
                
                group.attrs["composer"] = row["canonical_composer"]
                group.attrs["title"]    = row["canonical_title"]
                group.attrs["split"]    = row["split"]
                
                tqdm.write(f"Processed: {row['canonical_composer']} {row['canonical_title']}")
                
            print(f"Finished processing! Dataset saved at {self.f_out}")


if __name__=="__main__":
    dir_maestro = "dataset/maestro-v3.0.0"
    with open("config.json", "r") as f:
        config = json.load(f)

    prep_midr = MidrPreprocessor(config)
    prep_midi = MidiPreprocessor(config)
    prep_wav  = WavPreprocessor(config)
    processor = MaestroPreprocessor(dir_maestro, prep_wav, prep_midi, prep_midr)
    processor.process_all()