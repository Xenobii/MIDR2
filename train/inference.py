import torch
import json
from argparse import ArgumentParser

from model.model import Model2
from preprocessors.prep_wav import WavPreprocessor
from preprocessors.prep_midr import MidrPreprocessor


def run_inference(config, path):
    # Load checkpoint
    checkpoint_path = f"model/{path}/checkpoint_4.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Load model
    model = Model2(config)
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
    demo_input = torch.from_numpy(input[5]).to(device).unsqueeze(0)

    # Ablation
    prep_midr = MidrPreprocessor(config)
    midr = prep_midr(f_mid)
    # print(midr['spiral_cc'])
    # breakpoint()
    prep_midr.plot_chroma(midr['spiral_cc'][6].T)

    # print(demo_input.shape)
    # breakpoint()

    with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.float16):
        output_spiral_cd, output_spiral_cc = model(demo_input)
        print(output_spiral_cc.shape)
    prep_midr.plot_chroma(output_spiral_cc.squeeze(0).T.cpu())

    return



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', default='model')
    args = parser.parse_args()
    
    with open('config.json') as f:
        config = json.load(f)

    run_inference(config, args.path)