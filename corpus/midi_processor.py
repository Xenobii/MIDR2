import mido
import json
import numpy as np
import torch


class MIDIProcessor():
    def __init__(self, config):
        # midi
        self.num_pitch  = config['midi']['num_pitch']
        self.note_min   = config['midi']['note_min']
        self.note_max   = config['midi']['note_max']
        self.num_notes  = config['midi']['num_notes']

        # feature
        self.sr         = config['feature']['sr']
        self.hop_sample = config['feature']['hop_sample']

    def midi2note(self, f_midi, verbose_flag = False):
        # (1) read MIDI file
        midi_file = mido.MidiFile(f_midi)
        ticks_per_beat = midi_file.ticks_per_beat
        num_tracks = len(midi_file.tracks)

        # (2) tempo curve
        max_ticks_total = 0
        for it in range(len(midi_file.tracks)):
            ticks_total = 0
            for message in midi_file.tracks[it]:
                ticks_total += int(message.time)
            if max_ticks_total < ticks_total:
                max_ticks_total = ticks_total
        a_time_in_sec = [0.0 for i in range(max_ticks_total+1)]
        ticks_curr = 0
        ticks_prev = 0
        tempo_curr = 0
        tempo_prev = 0
        time_in_sec_prev = 0.0
        for im, message in enumerate(midi_file.tracks[0]):
            ticks_curr += message.time
            if 'set_tempo' in str(message):
                tempo_curr = int(message.tempo)
                for i in range(ticks_prev, ticks_curr):
                    a_time_in_sec[i] = time_in_sec_prev + ((i-ticks_prev) / ticks_per_beat * tempo_prev / 1e06)
                if ticks_curr > 0:
                    time_in_sec_prev = time_in_sec_prev + ((ticks_curr-ticks_prev) / ticks_per_beat * tempo_prev / 1e06)
                tempo_prev = tempo_curr
                ticks_prev = ticks_curr
        for i in range(ticks_prev, max_ticks_total+1):
            a_time_in_sec[i] = time_in_sec_prev + ((i-ticks_prev) / ticks_per_beat * tempo_curr / 1e06)

        # (3) obtain MIDI message
        a_note = []
        a_onset = []
        a_velocity = []
        a_reonset = []
        a_push = []
        a_sustain = []
        for i in range(self.num_pitch):
            a_onset.append(-1)
            a_velocity.append(-1)
            a_reonset.append(False)
            a_push.append(False)
            a_sustain.append(False)

        ticks = 0
        sustain_flag = False
        for message in midi_file.tracks[num_tracks-1]:
            ticks += message.time
            time_in_sec = a_time_in_sec[ticks]
            if verbose_flag is True:
                #print('[message]'+str(message)+' [ticks]: '+str(ticks/ticks_per_sec))
                print('[message]'+str(message)+' [ticks]: '+str(time_in_sec)+' [time]: '+str(time_in_sec))
            if ('control_change' in str(message)) and ('control=64' in str(message)):
                if message.value < 64:
                    # sustain off
                    if verbose_flag is True:
                        print('** sustain pedal OFF **')
                    for i in range(self.note_min, self.note_max+1):
                        if (a_push[i] is False) and (a_sustain[i] is True):
                            if verbose_flag is True:
                                print('## output sustain pedal off : '+str(i))
                                print({'onset': a_onset[i],
                                    'offset': time_in_sec,
                                    'pitch': i,
                                    'velocity': a_velocity[i],
                                    'reonset': a_reonset[i]})
                            a_note.append({'onset': a_onset[i],
                                        'offset': time_in_sec,
                                        'pitch': i,
                                        'velocity': a_velocity[i],
                                        'reonset': a_reonset[i]})
                            a_onset[i] = -1
                            a_velocity[i] = -1
                            a_reonset[i] = False
                    sustain_flag = False
                    for i in range(self.note_min, self.note_max+1):
                        a_sustain[i] = False
                else:
                    # sustain on
                    if verbose_flag is True:
                        print('** sustain pedal ON **')
                    sustain_flag = True
                    for i in range(self.note_min, self.note_max+1):
                        if a_push[i] is True:
                            a_sustain[i] = True
                            if verbose_flag is True:
                                print('sustain('+str(i)+') ON')
            elif ('note_on' in str(message)) and (int(message.velocity) > 0):
                # note on
                note = message.note
                velocity = message.velocity
                if verbose_flag is True:
                    print('++note ON++: '+str(note))
                if (a_push[note] is True) or (a_sustain[note] is True):
                    if verbose_flag is True:
                        print('## output reonset : '+str(note))
                        print({'onset': a_onset[note],
                            'offset': time_in_sec,
                            'pitch': note,
                            'velocity': a_velocity[note],
                            'reonset': a_reonset[note]})
                    # reonset
                    a_note.append({'onset': a_onset[note],
                                'offset': time_in_sec,
                                'pitch': note,
                                'velocity': a_velocity[note],
                                'reonset': a_reonset[note]})
                    a_reonset[note] = True
                else:
                    a_reonset[note] = False
                a_onset[note] = time_in_sec
                a_velocity[note] = velocity
                a_push[note] = True
                if sustain_flag is True:
                    a_sustain[note] = True
                    if verbose_flag is True:
                        print('sustain('+str(note)+') ON')
            elif (('note_off' in str(message)) or \
                (('note_on' in str(message)) and (int(message.velocity) == 0))):
                # note off
                note = message.note
                velocity = message.velocity
                if verbose_flag is True:
                    print('++note OFF++: '+str(note))
                if (a_push[note] is True) and (a_sustain[note] is False):
                    # offset
                    if verbose_flag is True:
                        print('## output offset : '+str(note))
                        print({'onset': a_onset[note],
                            'offset': time_in_sec,
                            'pitch': note,
                            'velocity': a_velocity[note],
                            'reonset': a_reonset[note]})
                        print({'onset': a_onset[note],
                            'offset': time_in_sec,
                            'pitch': note,
                            'velocity': a_velocity[note],
                            'reonset': a_reonset[note]})
                    a_note.append({'onset': a_onset[note],
                                'offset': time_in_sec,
                                'pitch': note,
                                'velocity': a_velocity[note],
                                'reonset': a_reonset[note]})
                    a_onset[note] = -1
                    a_velocity[note] = -1
                    a_reonset[note] = False
                a_push[note] = False

        for i in range(self.note_min, self.note_max+1):
            if (a_push[i] is True) or (a_sustain[i] is True):
                if verbose_flag is True:
                    print('## output final : '+str(i))
                    print({'onset': a_onset[i],
                        'offset': time_in_sec,
                        'pitch': i,
                        'velocity': a_velocity[i],
                        'reonset': a_reonset[i]})
                a_note.append({'onset': a_onset[i],
                            'offset': time_in_sec,
                            'pitch': i,
                            'velocity': a_velocity[i],
                            'reonset': a_reonset[i]})
        a_note_sort = sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['onset'])

        return a_note_sort
    
    def note2label(self, f_note, offset_duration_tolerance_flag=False):
        # (0) settings
        # tolerance: 50[ms]
        hop_ms = 1000 * self.hop_sample / self.sr
        onset_tolerance = int(50.0 / hop_ms + 0.5)
        offset_tolerance = int(50.0 / hop_ms + 0.5)

        with open(f_note, 'r', encoding='utf-8') as f:
            a_note = json.load(f)

        # 62.5 (hop=256, fs=16000)
        nframe_in_sec = self.sr / self.hop_sample

        max_offset = 0
        for note in a_note:
            if max_offset < note['offset']:
                max_offset = note['offset']
        
        nframe = int(max_offset * nframe_in_sec + 0.5) + 1
        a_mpe = np.zeros((nframe, self.num_notes), dtype=np.bool)
        a_onset = np.zeros((nframe, self.num_notes), dtype=np.float32)
        a_offset = np.zeros((nframe, self.num_notes), dtype=np.float32)
        a_velocity = np.zeros((nframe, self.num_notes), dtype=np.int8)

        for i in range(len(a_note)):
            pitch = a_note[i]['pitch'] - self.note_min

            # a_note[i]['onset'] in sec
            onset_frame = int(a_note[i]['onset'] * nframe_in_sec + 0.5)
            onset_ms = a_note[i]['onset']*1000.0
            onset_sharpness = onset_tolerance

            # a_note[i]['offset'] in sec
            offset_frame = int(a_note[i]['offset'] * nframe_in_sec + 0.5)
            offset_ms = a_note[i]['offset']*1000.0
            offset_sharpness = offset_tolerance

            if offset_duration_tolerance_flag is True:
                offset_duration_tolerance = int((offset_ms - onset_ms) * 0.2 / hop_ms + 0.5)
                offset_sharpness = max(offset_tolerance, offset_duration_tolerance)

            # velocity
            velocity = a_note[i]['velocity']

            # onset
            for j in range(0, onset_sharpness+1):
                onset_ms_q = (onset_frame + j) * hop_ms
                onset_ms_diff = onset_ms_q - onset_ms
                onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                if onset_frame+j < nframe:
                    a_onset[onset_frame+j][pitch] = max(a_onset[onset_frame+j][pitch], onset_val)
                    if (a_onset[onset_frame+j][pitch] >= 0.5):
                        a_velocity[onset_frame+j][pitch] = velocity

            for j in range(1, onset_sharpness+1):
                onset_ms_q = (onset_frame - j) * hop_ms
                onset_ms_diff = onset_ms_q - onset_ms
                onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                if onset_frame-j >= 0:
                    a_onset[onset_frame-j][pitch] = max(a_onset[onset_frame-j][pitch], onset_val)
                    if (a_onset[onset_frame-j][pitch] >= 0.5) and (a_velocity[onset_frame-j][pitch] == 0):
                        a_velocity[onset_frame-j][pitch] = velocity

            # mpe
            for j in range(onset_frame, offset_frame+1):
                a_mpe[j][pitch] = 1

            # offset
            offset_flag = True
            for j in range(len(a_note)):
                if a_note[i]['pitch'] != a_note[j]['pitch']:
                    continue
                if a_note[i]['offset'] == a_note[j]['onset']:
                    offset_flag = False
                    break

            if offset_flag is True:
                for j in range(0, offset_sharpness+1):
                    offset_ms_q = (offset_frame + j) * hop_ms
                    offset_ms_diff = offset_ms_q - offset_ms
                    offset_val = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                    if offset_frame+j < nframe:
                        a_offset[offset_frame+j][pitch] = max(a_offset[offset_frame+j][pitch], offset_val)
                for j in range(1, offset_sharpness+1):
                    offset_ms_q = (offset_frame - j) * hop_ms
                    offset_ms_diff = offset_ms_q - offset_ms
                    offset_val = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                    if offset_frame-j >= 0:
                        a_offset[offset_frame-j][pitch] = max(a_offset[offset_frame-j][pitch],  offset_val)
                        
        # (5-2) output label file
        # mpe        : 0 or 1
        # onset      : 0.0-1.0
        # offset     : 0.0-1.0
        # velocity   : 0 - 127
        a_label = {
            'mpe'       : a_mpe.tolist(),
            'onset'     : a_onset.tolist(),
            'offset'    : a_offset.tolist(),
            'velocity'  : a_velocity.tolist()
        }

        return a_label
    
    def note2ref(self, note_txt_path):
        def note2freq(note_number):
            return 440.0 * pow(2.0, (int(note_number) - 69) / 12)

        # Load note file (skip header)
        with open(note_txt_path, 'r', encoding='utf-8') as f:
            a_input = f.readlines()

        # Compute total duration
        duration = 0.0
        for i in range(1, len(a_input)):
            offset = float(a_input[i].rstrip('\n').split('\t')[1])
            if duration < offset:
                duration = offset

        # Frame counts for 16ms and 10ms
        nframe_16ms = int(duration * 62.5 + 0.5) + 1
        nframe_10ms = int(duration * 100 + 0.5) + 1

        # Binary pitch activations
        a_mpe_16 = np.zeros((nframe_16ms, self.num_pitch), dtype=np.int_)
        a_mpe_10 = np.zeros((nframe_10ms, self.num_pitch), dtype=np.int_)

        # Fill activation matrices
        for n in range(1, len(a_input)):
            cols = a_input[n].rstrip('\n').split('\t')
            onset = float(cols[0])
            offset = float(cols[1])
            pitch = int(cols[3])

            # 16 ms
            onset_frame = int(onset * 62.5 + 0.5)
            offset_frame = int(offset * 62.5 + 0.5)
            for i in range(onset_frame, offset_frame + 1):
                a_mpe_16[i][pitch] = 1

            # 10 ms
            onset_frame = int(onset * 100 + 0.5)
            offset_frame = int(offset * 100 + 0.5)
            for i in range(onset_frame, offset_frame + 1):
                a_mpe_10[i][pitch] = 1

        # Convert active MIDI pitches to frequencies per frame
        a_ref_16 = []
        a_ref_10 = []

        for i in range(len(a_mpe_16)):
            freqs = [note2freq(j) for j in range(self.num_pitch) if a_mpe_16[i][j] == 1]
            a_ref_16.append(freqs)

        for i in range(len(a_mpe_10)):
            freqs = [note2freq(j) for j in range(self.num_pitch) if a_mpe_10[i][j] == 1]
            a_ref_10.append(freqs)

        return a_ref_10, a_ref_16