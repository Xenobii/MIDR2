import os
import json
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

from preprocessors.prep_wav import WavPreprocessor
from preprocessors.prep_midr import MidrPreprocessor


class MaestroPreprocessor:
    def __init__(self, dir_maestro, prep_wav, prep_midr):
        self.maestro_dir = dir_maestro
        self.df          = pd.read_csv(os.path.join(self.maestro_dir, "maestro-v3.0.0_demo.csv"))
        self.f_out       = "dataset/processed_dataset.h5"
        self.prep_midr   = prep_midr
        self.prep_wav    = prep_wav

    def process_all(self):
        with h5py.File(self.f_out, "w") as h5:
            print(f"Processing maestro v3... \n")
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
                midi_path = os.path.join(self.maestro_dir, row["midi_filename"])
                wav_path  = os.path.join(self.maestro_dir, row["audio_filename"])

                spec_chunks = self.prep_wav(wav_path)
                midr_chunks = self.prep_midr(midi_path)
                print(len(spec_chunks))
                print(len(midr_chunks["spiral_cc"]))
                breakpoint()
                assert len(spec_chunks) == len(midr_chunks["spiral_cc"]), \
                    f"Chunk missmatch: {len(spec_chunks) - len(midr_chunks['spiral_cc'])}"

                group = h5.create_group(f"{idx:07d}")
                group.create_dataset("spec", data=spec_chunks, compression="lzf")
                group.create_dataset("circle_cd", data=midr_chunks["circle_cd"], compression="lzf")
                group.create_dataset("circle_cc", data=midr_chunks["circle_cc"], compression="lzf")
                group.create_dataset("spiral_cd", data=midr_chunks["spiral_cd"], compression="lzf")
                group.create_dataset("spiral_cc", data=midr_chunks["spiral_cc"], compression="lzf")
                
                # group.create_dataset("midr", data=midr_chunks, compression="lzf")
                group.attrs["composer"] = row["canonical_composer"]
                group.attrs["title"]    = row["canonical_title"]
                group.attrs["split"]    = row["split"]
                
                print(f"Processed: {row['canonical_title']}")
                
            print(f"Finished processing! Dataset saved at {self.f_out}")


if __name__=="__main__":
    dir_maestro = "dataset/maestro-v3.0.0"
    with open("config.json", "r") as f:
        config = json.load(f)

    prep_midr = MidrPreprocessor(config)
    prep_wav = WavPreprocessor(config)
    processor = MaestroPreprocessor(dir_maestro, prep_wav, prep_midr)
    processor.process_all()