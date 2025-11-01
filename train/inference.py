import torch
import json
from argparse import ArgumentParser

from model.model import AMT_1
from preprocessors.prep_wav import WavPreprocessor
from preprocessors.prep_midr import MidrPreprocessor
from preprocessors.prep_midi import MidiPreprocessor


def run_inference(config, path):
    # Load checkpoint
    checkpoint_path = f"model/{path}/checkpoint_4.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Load model
    model = AMT_1(config)
    model.load_state_dict(checkpoint['model_dict'])
    
    # Set to eval mode
    model.eval()

    # Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Get demo input
    f_wav = "test_files/test_wav.WAV"
    f_mid = "test_files/test_midi.MID"
        
    # Preprocess
    prep_wav = WavPreprocessor(config)
    input = prep_wav(f_wav)
    demo_input = torch.from_numpy(input[0]).to(device).unsqueeze(0)

    # Ablation
    prep_midr = MidrPreprocessor(config)

    prep_midi = MidiPreprocessor(config)

    with torch.no_grad():
        # output_spiral_cd, output_spiral_cc = model(demo_input)
        output_mpe, output_onset, output_offset, output_velocity = model(demo_input)
        print(output_mpe.shape)
    prep_midi.plot_midi(output_mpe.squeeze(0).cpu())
    prep_midi.plot_midi(output_onset.squeeze(0).cpu())
    prep_midi.plot_midi(output_offset.squeeze(0).cpu())
    # prep_midi.plot_midi(output_velocity.squeeze(0).T.cpu())

    return



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', default='model')
    args = parser.parse_args()
    
    with open('config.json') as f:
        config = json.load(f)

    run_inference(config, args.path)