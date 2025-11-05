import mir_eval
import pretty_midi
import numpy as np


# ALL CLAUDE CAUSE I CANT BE FUCKED TO DO THIS SHIT


class MIDIEvaluator:
    """
    Evaluator class for computing mir_eval metrics between reference and predicted MIDI files.
    Supports frame-level and note-level evaluation metrics.
    """
    
    def __init__(self, hop_length=512, sr=16000, remove_first_measure=True):
        """
        Initialize the evaluator.
        
        Args:
            hop_length: Hop length in samples for frame-based metrics
            sr: Sample rate in Hz
            remove_first_measure: If True, remove the first 4 beats from reference MIDI (default: True)
        """
        self.hop_length = hop_length
        self.sr = sr
        self.hop_time = hop_length / sr
        self.remove_first_measure = remove_first_measure
        
    def midi_to_intervals_pitches(self, midi_file):
        """
        Convert MIDI file to intervals and pitches format for mir_eval.
        
        Args:
            midi_file: Path to MIDI file or PrettyMIDI object
            
        Returns:
            intervals: numpy array of shape (n_notes, 2) with [onset, offset] times
            pitches: numpy array of shape (n_notes,) with MIDI pitch numbers
            velocities: numpy array of shape (n_notes,) with velocities
        """
        if isinstance(midi_file, str):
            midi_data = pretty_midi.PrettyMIDI(midi_file)
        else:
            midi_data = midi_file
            
        intervals = []
        pitches = []
        velocities = []
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    intervals.append([note.start, note.end])
                    pitches.append(note.pitch)
                    velocities.append(note.velocity)
        
        if len(intervals) == 0:
            return np.array([]).reshape(0, 2), np.array([]), np.array([])
        
        intervals = np.array(intervals)
        pitches = np.array(pitches)
        velocities = np.array(velocities)
        
        # Sort by onset time
        sort_idx = np.argsort(intervals[:, 0])
        intervals = intervals[sort_idx]
        pitches = pitches[sort_idx]
        velocities = velocities[sort_idx]
        
        return intervals, pitches, velocities
    
    def remove_first_measure_from_midi(self, midi_file, beats_to_remove=4):
        """
        Remove the first measure (first N beats) from a MIDI file.
        
        Args:
            midi_file: Path to MIDI file or PrettyMIDI object
            beats_to_remove: Number of beats to remove from the beginning (default: 4)
            
        Returns:
            PrettyMIDI object with first measure removed
        """
        if isinstance(midi_file, str):
            midi_data = pretty_midi.PrettyMIDI(midi_file)
        else:
            midi_data = midi_file
        
        # Estimate tempo and calculate time duration of first measure
        tempo = midi_data.estimate_tempo()
        beat_duration = 60.0 / tempo  # duration of one beat in seconds
        measure_duration = beat_duration * beats_to_remove
        
        # Create new MIDI object
        new_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Copy instruments and shift notes
        for instrument in midi_data.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )
            
            # Add notes that start after the first measure, shifted back in time
            for note in instrument.notes:
                if note.start >= measure_duration:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start - measure_duration,
                        end=note.end - measure_duration
                    )
                    new_instrument.notes.append(new_note)
            
            new_midi.instruments.append(new_instrument)
        
        return new_midi
    
    def preprocess_midi(self, midi_file):
        """
        Preprocess MIDI file (e.g., remove first measure if needed).
        
        Args:
            midi_file: Path to MIDI file or PrettyMIDI object
            
        Returns:
            PrettyMIDI object or tuple of (intervals, pitches, velocities)
        """
        if self.remove_first_measure:
            return self.remove_first_measure_from_midi(midi_file)
        else:
            if isinstance(midi_file, str):
                return pretty_midi.PrettyMIDI(midi_file)
            return midi_file
    
    def midi_to_piano_roll(self, midi_file, end_time=None):
        """
        Convert MIDI file to piano roll representation.
        
        Args:
            midi_file: Path to MIDI file or PrettyMIDI object
            end_time: End time in seconds (if None, use the end of last note)
            
        Returns:
            piano_roll: numpy array of shape (128, n_frames) with binary activations
        """
        if isinstance(midi_file, str):
            midi_data = pretty_midi.PrettyMIDI(midi_file)
        else:
            midi_data = midi_file
            
        if end_time is None:
            end_time = midi_data.get_end_time()
            
        # Get piano roll with the specified hop time
        piano_roll = midi_data.get_piano_roll(fs=1.0/self.hop_time)
        
        # Convert to binary
        piano_roll = (piano_roll > 0).astype(float)
        
        return piano_roll
    
    def compute_frame_metrics(self, ref_midi, est_midi):
        """
        Compute frame-level metrics (F-measure, precision, recall).
        Uses piano roll representation for true frame-level evaluation.
        
        Args:
            ref_midi: Path to reference MIDI file
            est_midi: Path to estimated MIDI file
            
        Returns:
            dict: Dictionary containing frame metrics
        """
        # Preprocess both MIDI files
        ref_midi = self.preprocess_midi(ref_midi)
        est_midi = self.preprocess_midi(est_midi)
        
        # Convert to piano rolls
        ref_roll = self.midi_to_piano_roll(ref_midi)
        est_roll = self.midi_to_piano_roll(est_midi)
        
        # Make sure both piano rolls have the same length
        max_frames = max(ref_roll.shape[1], est_roll.shape[1])
        if ref_roll.shape[1] < max_frames:
            ref_roll = np.pad(ref_roll, ((0, 0), (0, max_frames - ref_roll.shape[1])), mode='constant')
        if est_roll.shape[1] < max_frames:
            est_roll = np.pad(est_roll, ((0, 0), (0, max_frames - est_roll.shape[1])), mode='constant')
        
        # Flatten to get frame-level predictions
        ref_frames = ref_roll.T  # (n_frames, 128)
        est_frames = est_roll.T  # (n_frames, 128)
        
        # Compute frame-level metrics
        # For each frame, check if the active pitches match
        ref_active = ref_frames.sum(axis=1) > 0  # frames with any active notes
        est_active = est_frames.sum(axis=1) > 0
        
        # True positives: frames where both ref and est have the same active pitches
        tp = np.sum(np.all(ref_frames == est_frames, axis=1) & (ref_active | est_active))
        
        # All frames with activity
        total_ref = np.sum(ref_active)
        total_est = np.sum(est_active)
        
        # Use mir_eval's multipitch metrics for proper frame-level evaluation
        ref_times = np.arange(ref_frames.shape[0]) * self.hop_time
        est_times = np.arange(est_frames.shape[0]) * self.hop_time
        
        # Convert to list of active pitches per frame (as numpy arrays)
        ref_pitch_list = [np.where(frame > 0)[0] for frame in ref_frames]
        est_pitch_list = [np.where(frame > 0)[0] for frame in est_frames]
        
        # Use mir_eval's multipitch evaluation
        scores = mir_eval.multipitch.evaluate(ref_times, ref_pitch_list, est_times, est_pitch_list)
        
        return {
            'Frame Precision': scores['Precision'],
            'Frame Recall': scores['Recall'],
            'Frame F-measure': scores['Accuracy'],
        }
    
    def compute_note_metrics(self, ref_midi, est_midi, onset_tolerance=0.05, offset_ratio=0.2):
        """
        Compute note-level metrics with and without offsets.
        
        Args:
            ref_midi: Path to reference MIDI file
            est_midi: Path to estimated MIDI file
            onset_tolerance: Onset tolerance in seconds (default: 50ms)
            offset_ratio: Offset tolerance as ratio of note duration (default: 0.2)
            
        Returns:
            dict: Dictionary containing note metrics
        """
        # Preprocess both MIDI files
        ref_midi = self.preprocess_midi(ref_midi)
        est_midi = self.preprocess_midi(est_midi)
        
        ref_intervals, ref_pitches, _ = self.midi_to_intervals_pitches(ref_midi)
        est_intervals, est_pitches, _ = self.midi_to_intervals_pitches(est_midi)
        
        metrics = {}
        
        # Note with onset only (F0)
        precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=0.5,
            offset_ratio=None
        )
        metrics['Note Precision (onset only)'] = precision
        metrics['Note Recall (onset only)'] = recall
        metrics['Note F-measure (onset only)'] = f_measure
        
        # Note with onset and offset (FNO)
        precision_offset, recall_offset, f_measure_offset, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=0.5,
            offset_ratio=offset_ratio,
            offset_min_tolerance=0.05
        )
        metrics['Note-with-offset Precision'] = precision_offset
        metrics['Note-with-offset Recall'] = recall_offset
        metrics['Note-with-offset F-measure'] = f_measure_offset
        
        return metrics
    
    def compute_velocity_metrics(self, ref_midi, est_midi, onset_tolerance=0.05):
        """
        Compute velocity-related metrics.
        
        Args:
            ref_midi: Path to reference MIDI file
            est_midi: Path to estimated MIDI file
            onset_tolerance: Onset tolerance in seconds for matching notes
            
        Returns:
            dict: Dictionary containing velocity metrics
        """
        # Preprocess both MIDI files
        ref_midi = self.preprocess_midi(ref_midi)
        est_midi = self.preprocess_midi(est_midi)
        
        ref_intervals, ref_pitches, ref_velocities = self.midi_to_intervals_pitches(ref_midi)
        est_intervals, est_pitches, est_velocities = self.midi_to_intervals_pitches(est_midi)
        
        # Match notes based on onset and pitch
        matched_velocities_ref = []
        matched_velocities_est = []
        
        for i, (ref_int, ref_pitch) in enumerate(zip(ref_intervals, ref_pitches)):
            for j, (est_int, est_pitch) in enumerate(zip(est_intervals, est_pitches)):
                # Check if onset is within tolerance and pitch matches
                if abs(ref_int[0] - est_int[0]) <= onset_tolerance and ref_pitch == est_pitch:
                    matched_velocities_ref.append(ref_velocities[i])
                    matched_velocities_est.append(est_velocities[j])
                    break
        
        if len(matched_velocities_ref) > 0:
            matched_velocities_ref = np.array(matched_velocities_ref)
            matched_velocities_est = np.array(matched_velocities_est)
            
            # Compute velocity metrics
            vel_mae = np.mean(np.abs(matched_velocities_ref - matched_velocities_est))
            vel_rmse = np.sqrt(np.mean((matched_velocities_ref - matched_velocities_est) ** 2))
            
            return {
                'Velocity MAE': vel_mae,
                'Velocity RMSE': vel_rmse,
                'Matched Notes': len(matched_velocities_ref)
            }
        else:
            return {
                'Velocity MAE': float('nan'),
                'Velocity RMSE': float('nan'),
                'Matched Notes': 0
            }
    
    def evaluate(self, ref_midi, est_midi, verbose=True):
        """
        Compute all evaluation metrics.
        
        Args:
            ref_midi: Path to reference MIDI file
            est_midi: Path to estimated MIDI file
            verbose: If True, print metrics
            
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {}
        
        # Frame-level metrics
        frame_metrics = self.compute_frame_metrics(ref_midi, est_midi)
        metrics.update(frame_metrics)
        
        # Note-level metrics
        note_metrics = self.compute_note_metrics(ref_midi, est_midi)
        metrics.update(note_metrics)
        
        # Velocity metrics
        velocity_metrics = self.compute_velocity_metrics(ref_midi, est_midi)
        metrics.update(velocity_metrics)
        
        if verbose:
            self.print_metrics(metrics)
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("MIDI Transcription Evaluation Metrics")
        print("="*60)
        
        print("\n--- Frame-level Metrics ---")
        print(f"  Precision: {metrics['Frame Precision']:.4f}")
        print(f"  Recall:    {metrics['Frame Recall']:.4f}")
        print(f"  F-measure: {metrics['Frame F-measure']:.4f}")
        
        print("\n--- Note-level Metrics (Onset Only) ---")
        print(f"  Precision: {metrics['Note Precision (onset only)']:.4f}")
        print(f"  Recall:    {metrics['Note Recall (onset only)']:.4f}")
        print(f"  F-measure: {metrics['Note F-measure (onset only)']:.4f}")
        
        print("\n--- Note-level Metrics (Onset + Offset) ---")
        print(f"  Precision: {metrics['Note-with-offset Precision']:.4f}")
        print(f"  Recall:    {metrics['Note-with-offset Recall']:.4f}")
        print(f"  F-measure: {metrics['Note-with-offset F-measure']:.4f}")
        
        print("\n--- Velocity Metrics ---")
        if metrics['Matched Notes'] > 0:
            print(f"  MAE:           {metrics['Velocity MAE']:.2f}")
            print(f"  RMSE:          {metrics['Velocity RMSE']:.2f}")
            print(f"  Matched Notes: {metrics['Matched Notes']}")
        else:
            print("  No matched notes found")
        
        print("="*60 + "\n")
    
    def __call__(self, ref_midi, est_midi):
        """
        Evaluate and print metrics when instance is called.
        
        Args:
            ref_midi: Path to reference MIDI file
            est_midi: Path to estimated MIDI file
            
        Returns:
            dict: Dictionary containing all metrics
        """
        return self.evaluate(ref_midi, est_midi, verbose=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MIDI transcription')
    parser.add_argument('-r', '--reference', required=True, help='Path to reference MIDI file')
    parser.add_argument('-e', '--estimated', required=True, help='Path to estimated MIDI file')
    parser.add_argument('--no-remove-first-measure', action='store_true', 
                        help='Do not remove first measure from reference MIDI')
    args = parser.parse_args()
    
    evaluator = MIDIEvaluator(remove_first_measure=not args.no_remove_first_measure)
    evaluator(args.reference, args.estimated)