import torch
import numpy as np
import pretty_midi
import json
from argparse import ArgumentParser

from preprocessors.prep_wav import WavPreprocessor
from model.model import AMT_1



class AMT():
    def __init__(self, config, model_path, verbose=False):
        if verbose is True:
            print(f"torch version   : {torch.__version__}")
            print(f"torch cuda      : {str(torch.cuda.is_available())}")
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load checkpoint
        checkpoint_path = f"model/{model_path}/checkpoint_1.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Load model
        self.model = AMT_1(config)
        self.model.load_state_dict(checkpoint['model_dict'])
        self.model.eval()

        # Move to device
        self.model = self.model.to(self.device)

    def wav2feat(self, f_wav):
        prep_wav = WavPreprocessor(self.config)
        return prep_wav(f_wav)

    def feat2mpe(self, feat) -> torch.Tensor:
        feat = torch.from_numpy(feat).to(self.device, non_blocking=True)
        a_mpe, a_onset, a_offset, a_velocity = self.model(feat)
        
        # concat across batch (chunk) dimension
        chunks, nframe, nbin = a_mpe.shape
        a_mpe      = a_mpe.reshape(chunks*nframe, nbin).to('cpu').detach().numpy()
        a_onset    = a_onset.reshape(chunks*nframe, nbin).to('cpu').detach().numpy()
        a_offset   = a_offset.reshape(chunks*nframe, nbin).to('cpu').detach().numpy()
        a_velocity = a_velocity.reshape(chunks*nframe, nbin, 128).argmax(1).to('cpu').detach().numpy()

        return a_mpe, a_onset, a_offset, a_velocity
    

    def mpe2note(self, a_onset=None, a_offset=None, a_mpe=None, a_velocity=None, thred_onset=0.5, thred_offset=0.5, thred_mpe=0.5, mode_velocity='ignore_zero', mode_offset='shorter'):
        # mode_velocity
        #  org: 0-127
        #  ignore_zero: 0-127 (output note does not include 0) (default)

        # mode_offset
        #  shorter: use shorter one of mpe and offset (default)
        #  longer : use longer one of mpe and offset
        #  offset : use offset (ignore mpe)

        a_note = []
        hop_sec = float(self.config['spec']['hop_sample'] / self.config['spec']['sr'])

        for j in range(self.config['midi']['num_notes']):
            # find local maximum
            a_onset_detect = []
            for i in range(len(a_onset)):
                if a_onset[i][j] >= thred_onset:
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if a_onset[i][j] > a_onset[ii][j]:
                            left_flag = True
                            break
                        elif a_onset[i][j] < a_onset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(a_onset)):
                        if a_onset[i][j] > a_onset[ii][j]:
                            right_flag = True
                            break
                        elif a_onset[i][j] < a_onset[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):
                        if (i == 0) or (i == len(a_onset) - 1):
                            onset_time = i * hop_sec
                        else:
                            if a_onset[i-1][j] == a_onset[i+1][j]:
                                onset_time = i * hop_sec
                            elif a_onset[i-1][j] > a_onset[i+1][j]:
                                onset_time = (i * hop_sec - (hop_sec * 0.5 * (a_onset[i-1][j] - a_onset[i+1][j]) / (a_onset[i][j] - a_onset[i+1][j])))
                            else:
                                onset_time = (i * hop_sec + (hop_sec * 0.5 * (a_onset[i+1][j] - a_onset[i-1][j]) / (a_onset[i][j] - a_onset[i-1][j])))
                        a_onset_detect.append({'loc': i, 'onset_time': onset_time})
            a_offset_detect = []
            for i in range(len(a_offset)):
                if a_offset[i][j] >= thred_offset:
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if a_offset[i][j] > a_offset[ii][j]:
                            left_flag = True
                            break
                        elif a_offset[i][j] < a_offset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(a_offset)):
                        if a_offset[i][j] > a_offset[ii][j]:
                            right_flag = True
                            break
                        elif a_offset[i][j] < a_offset[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):
                        if (i == 0) or (i == len(a_offset) - 1):
                            offset_time = i * hop_sec
                        else:
                            if a_offset[i-1][j] == a_offset[i+1][j]:
                                offset_time = i * hop_sec
                            elif a_offset[i-1][j] > a_offset[i+1][j]:
                                offset_time = (i * hop_sec - (hop_sec * 0.5 * (a_offset[i-1][j] - a_offset[i+1][j]) / (a_offset[i][j] - a_offset[i+1][j])))
                            else:
                                offset_time = (i * hop_sec + (hop_sec * 0.5 * (a_offset[i+1][j] - a_offset[i-1][j]) / (a_offset[i][j] - a_offset[i-1][j])))
                        a_offset_detect.append({'loc': i, 'offset_time': offset_time})

            time_next = 0.0
            time_offset = 0.0
            time_mpe = 0.0
            for idx_on in range(len(a_onset_detect)):
                # onset
                loc_onset = a_onset_detect[idx_on]['loc']
                time_onset = a_onset_detect[idx_on]['onset_time']

                if idx_on + 1 < len(a_onset_detect):
                    loc_next = a_onset_detect[idx_on+1]['loc']
                    #time_next = loc_next * hop_sec
                    time_next = a_onset_detect[idx_on+1]['onset_time']
                else:
                    loc_next = len(a_mpe)
                    time_next = (loc_next-1) * hop_sec

                # offset
                loc_offset = loc_onset+1
                flag_offset = False
                #time_offset = 0###
                for idx_off in range(len(a_offset_detect)):
                    if loc_onset < a_offset_detect[idx_off]['loc']:
                        loc_offset = a_offset_detect[idx_off]['loc']
                        time_offset = a_offset_detect[idx_off]['offset_time']
                        flag_offset = True
                        break
                if loc_offset > loc_next:
                    loc_offset = loc_next
                    time_offset = time_next

                # offset by MPE
                # (1frame longer)
                loc_mpe = loc_onset+1
                flag_mpe = False
                #time_mpe = 0###
                for ii_mpe in range(loc_onset+1, loc_next):
                    if a_mpe[ii_mpe][j] < thred_mpe:
                        loc_mpe = ii_mpe
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                '''
                # (right algorighm)
                loc_mpe = loc_onset
                flag_mpe = False
                for ii_mpe in range(loc_onset+1, loc_next+1):
                    if a_mpe[ii_mpe][j] < thred_mpe:
                        loc_mpe = ii_mpe-1
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                '''
                pitch_value = int(j+self.config['midi']['note_min'])
                velocity_value = int(a_velocity[loc_onset][j])

                if (flag_offset is False) and (flag_mpe is False):
                    offset_value = float(time_next)
                elif (flag_offset is True) and (flag_mpe is False):
                    offset_value = float(time_offset)
                elif (flag_offset is False) and (flag_mpe is True):
                    offset_value = float(time_mpe)
                else:
                    if mode_offset == 'offset':
                        ## (a) offset
                        offset_value = float(time_offset)
                    elif mode_offset == 'longer':
                        ## (b) longer
                        if loc_offset >= loc_mpe:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_mpe)
                    else:
                        ## (c) shorter
                        if loc_offset <= loc_mpe:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_mpe)
                if mode_velocity != 'ignore_zero':
                    a_note.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})
                else:
                    if velocity_value > 0:
                        a_note.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})

                if (len(a_note) > 1) and \
                   (a_note[len(a_note)-1]['pitch'] == a_note[len(a_note)-2]['pitch']) and \
                   (a_note[len(a_note)-1]['onset'] < a_note[len(a_note)-2]['offset']):
                    a_note[len(a_note)-2]['offset'] = a_note[len(a_note)-1]['onset']

        a_note = sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['onset'])
        return a_note


    def note2midi(self, a_note, f_midi):
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        for note in a_note:
            instrument.notes.append(pretty_midi.Note(velocity=note['velocity'], pitch=note['pitch'], start=note['onset'], end=note['offset']))
        midi.instruments.append(instrument)
        midi.write(f_midi)
        print(f"Wrote files at {f_midi}")

        return
        
    def __call__(self, f_path, f_midi='test_files/test_ouput.mid'):
        # Prep wav
        feat = self.wav2feat(f_path)
        # Run inference
        a_mpe, a_onset, a_offset, a_velocity = self.feat2mpe(feat)
        # Get note list
        a_note = self.mpe2note(a_onset, a_offset, a_mpe, a_velocity)
        # Create midi
        self.note2midi(a_note, f_midi)





if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', default='model')
    parser.add_argument('-p', '--wav_path', default='test_files/test_wav.WAV')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    
    amt = AMT(config, model_path=args.model_path)
    amt(args.wav_path)
