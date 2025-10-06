import json
import pickle
import argparse

from midi_processor import MIDIProcessor
from spec_processor import SpecProcessor


def create_x(config):
    processor = SpecProcessor(config)

    # Feat
    a_feature = processor.wav2feat('test_files/test_wav.WAV')
    with open('test_files/a_feat.pkl', 'wb') as f:
        pickle.dump(a_feature, f, protocol=4)


def create_y(config):
    processor = MIDIProcessor(config)
    
    # Note
    a_note = processor.midi2note('test_files/test_midi.MID')
    with open('test_files/a_note.json', 'w', encoding='utf-8') as f:
        json.dump(a_note, f, ensure_ascii=False, indent=4, sort_keys=False)
    with open('test_files/a_note.txt', 'w', encoding='utf-8') as f:
        f.write('OnsetTime\tOffsetTime\tVelocity\tMidiPitch\n')
        for note in a_note:
            f.write(str(note['onset'])+'\t')
            f.write(str(note['offset'])+'\t')
            f.write(str(note['velocity'])+'\t')
            f.write(str(note['pitch'])+'\n')

    # List
    a_label = processor.note2label('test_files/a_note.json')
    with open('test_files/a_label.pkl', 'wb') as f:
        pickle.dump(a_label, f, protocol=4)

    # Ref
    a_ref_10, a_ref_16 = processor.note2ref('test_files/a_note.txt')
    with open('test_files/a_mpe_10ms.txt', 'w', encoding='utf-8') as f10:
        for i, freqs in enumerate(a_ref_10):
            f10.write(f"{i * 0.01:.2f}")
            for f_val in freqs:
                f10.write(f"\t{f_val}")
            f10.write("\n")
    with open('test_files/a_mpe_16ms.txt', 'w', encoding='utf-8') as f16:
        for i, freqs in enumerate(a_ref_16):
            f16.write(f"{i * 0.016:.3f}")
            for f_val in freqs:
                f16.write(f"\t{f_val}")
            f16.write("\n")



if __name__=="__main__":
    # Config
    with open('config.json', encoding='utf-8') as f:
        config = json.load(f)

    create_x(config)
    # create_y(config)