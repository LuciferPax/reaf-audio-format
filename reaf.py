import struct
import zlib
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import sys
import json

BREAK_MARKER = 0xFFFFFFFF # Use this value to represent breaks in the audio

# qualities
CD = {
    "sample_rate": 44100,
    "bit_depth": 16
}

DVD = {
    "sample_rate": 48000,
    "bit_depth": 24
}

STUDIO = {
    "sample_rate": 200000,
    "bit_depth": 32
}

def create_custom_audio_file(frequencies, sample_rate, bit_depth, output_file, break_value=BREAK_MARKER, metadata=""):
    def encode_sample(sample):
        return struct.pack('f', sample)
    
    def apply_fade(wave, fade_duration, sample_rate):
        fade_samples = int(fade_duration * sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out
        return wave
    
    fade_duration = 0.01  # 10 ms fade in/out for smooth transitions
    waveform_data = []

    for left_freq, right_freq, duration in frequencies:
        num_samples = int(sample_rate * duration)
        time = np.linspace(0, duration, num_samples, endpoint=False)
        
        if left_freq == break_value and right_freq == break_value:
            # Use RLE for breaks
            waveform_data.append(encode_sample(float(break_value)))
            waveform_data.append(encode_sample(duration))
        else:
            left_wave = 0.5 * np.sin(2 * np.pi * left_freq * time) if left_freq != break_value else np.zeros(num_samples)
            right_wave = 0.5 * np.sin(2 * np.pi * right_freq * time) if right_freq != break_value else np.zeros(num_samples)
            left_wave = apply_fade(left_wave, fade_duration, sample_rate)
            right_wave = apply_fade(right_wave, fade_duration, sample_rate)
            for l_sample, r_sample in zip(left_wave, right_wave):
                waveform_data.append(encode_sample(l_sample))
                waveform_data.append(encode_sample(r_sample))
    
    # Compress the waveform data
    compressed_data = zlib.compress(b''.join(waveform_data))

    with open(output_file, 'wb') as f:
        # Calculate the total duration
        total_duration = sum(duration for _, _, duration in frequencies)
        
        # Write header
        f.write(b'REAF')  # Magic number
        f.write(struct.pack('B', 1))  # Version
        f.write(struct.pack('I', sample_rate))
        f.write(struct.pack('B', bit_depth))
        f.write(struct.pack('B', 2))  # Stereo channel
        f.write(struct.pack('f', total_duration))  # Total duration as 32-bit float
        f.write(struct.pack('f', float(break_value)))  # Break value as 32-bit float
        f.write(struct.pack('I', len(metadata)))  # Metadata length
        f.write(metadata.encode('utf-8'))  # Metadata
        
        # Write compressed waveform data length and data
        f.write(struct.pack('I', len(compressed_data)))
        f.write(compressed_data)

def play_custom_audio_file(input_file):
    def decode_sample(sample_data):
        return struct.unpack('f', sample_data)[0]
    
    with open(input_file, 'rb') as f:
        magic_number = f.read(4)
        if magic_number != b'REAF':
            raise ValueError("Invalid file format")
        
        version = struct.unpack('B', f.read(1))[0]
        sample_rate = struct.unpack('I', f.read(4))[0]
        bit_depth = struct.unpack('B', f.read(1))[0]
        channels = struct.unpack('B', f.read(1))[0]
        total_duration = struct.unpack('f', f.read(4))[0]  # Total duration as 32-bit float
        break_value = struct.unpack('f', f.read(4))[0]  # Break value as 32-bit float
        
        metadata_length = struct.unpack('I', f.read(4))[0]
        metadata = f.read(metadata_length).decode('utf-8')
        
        compressed_data_length = struct.unpack('I', f.read(4))[0]
        compressed_data = f.read(compressed_data_length)
        
        # Decompress the waveform data
        decompressed_data = zlib.decompress(compressed_data)
        
        waveform_data = []
        i = 0
        while i < len(decompressed_data):
            sample_data = decompressed_data[i:i+4]
            sample = decode_sample(sample_data)
            i += 4
            if sample == break_value:
                duration_data = decompressed_data[i:i+4]
                duration = decode_sample(duration_data)
                silence_samples = int(duration * sample_rate)
                waveform_data.extend([0.0, 0.0] * silence_samples)
                i += 4
            else:
                waveform_data.append(sample)
        
        waveform = np.array(waveform_data, dtype=np.float32).reshape(-1, 2)
        sd.play(waveform, samplerate=sample_rate)
        sd.wait()
    
    return metadata

def convert_to_mp3(input_file, output_file):
    def decode_sample(sample_data):
        return struct.unpack('f', sample_data)[0]

    with open(input_file, 'rb') as f:
        magic_number = f.read(4)
        if magic_number != b'REAF':
            raise ValueError("Invalid file format")
        
        version = struct.unpack('B', f.read(1))[0]
        sample_rate = struct.unpack('I', f.read(4))[0]
        bit_depth = struct.unpack('B', f.read(1))[0]
        channels = struct.unpack('B', f.read(1))[0]
        total_duration = struct.unpack('f', f.read(4))[0]  # Total duration as 32-bit float
        break_value = struct.unpack('f', f.read(4))[0]  # Break value as 32-bit float
        
        metadata_length = struct.unpack('I', f.read(4))[0]
        metadata = f.read(metadata_length).decode('utf-8')
        
        compressed_data_length = struct.unpack('I', f.read(4))[0]
        compressed_data = f.read(compressed_data_length)
        
        # Decompress the waveform data
        decompressed_data = zlib.decompress(compressed_data)
        
        waveform_data = []
        i = 0
        while i < len(decompressed_data):
            sample_data = decompressed_data[i:i+4]
            sample = decode_sample(sample_data)
            i += 4
            if sample == break_value:
                duration_data = decompressed_data[i:i+4]
                duration = decode_sample(duration_data)
                silence_samples = int(duration * sample_rate)
                waveform_data.extend([0.0, 0.0] * silence_samples)
                i += 4
            else:
                waveform_data.append(sample)
        
        waveform = np.array(waveform_data, dtype=np.float32).reshape(-1, 2)

    audio_segment = AudioSegment(
        waveform.tobytes(), 
        frame_rate=sample_rate,
        sample_width=bit_depth // 8,
        channels=channels
    )
    
    audio_segment.export(output_file, format='mp3')

def convert_from_mp3(input_file, output_file, break_value=BREAK_MARKER, metadata=""):
    audio_segment = AudioSegment.from_file(input_file, format='mp3')
    samples = np.array(audio_segment.get_array_of_samples())
    
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = np.column_stack((samples, samples))  # Mono to Stereo
    
    def encode_sample(sample):
        return struct.pack('f', sample)

    waveform_data = []
    for left_sample, right_sample in samples:
        waveform_data.append(encode_sample(left_sample / 32768.0))
        waveform_data.append(encode_sample(right_sample / 32768.0))
    
    compressed_data = zlib.compress(b''.join(waveform_data))

    with open(output_file, 'wb') as f:
        total_duration = len(samples) / audio_segment.frame_rate
        
        f.write(b'REAF')  # Magic number
        f.write(struct.pack('B', 1))  # Version
        f.write(struct.pack('I', audio_segment.frame_rate))
        f.write(struct.pack('B', 16))  # Assuming 16-bit depth
        f.write(struct.pack('B', 2))  # Stereo channel
        f.write(struct.pack('f', total_duration))  # Total duration as 32-bit float
        f.write(struct.pack('f', float(break_value)))  # Break value as 32-bit float
        f.write(struct.pack('I', len(metadata)))  # Metadata length
        f.write(metadata.encode('utf-8'))  # Metadata
        
        # Write compressed waveform data length and data
        f.write(struct.pack('I', len(compressed_data)))
        f.write(compressed_data)

def main(args):
    command = args[0]
    if command == 'create':
        frequencies = json.loads(args[1])
        quality = args[2]
        output_file = args[3]

        if not output_file.endswith('.reaf'):
            output_file += '.reaf'

        meta = args[4:]

        if quality == 'CD':
            create_custom_audio_file(frequencies, CD["sample_rate"], CD["bit_depth"], output_file, metadata=" ".join(meta))
        elif quality == 'DVD':
            create_custom_audio_file(frequencies, DVD["sample_rate"], DVD["bit_depth"], output_file, metadata=" ".join(meta))
        elif quality == 'STUDIO':
            create_custom_audio_file(frequencies, STUDIO["sample_rate"], STUDIO["bit_depth"], output_file, metadata=" ".join(meta))
    elif command == 'play':
        input_file = args[1]
        play_custom_audio_file(input_file)
    elif command == 'convert-to-mp3':
        input_file = args[1]
        output_file = args[2]

        if not output_file.endswith('.mp3'):
            output_file += '.mp3'

        convert_to_mp3(input_file, output_file)
    elif command == 'convert-from-mp3':
        input_file = args[1]
        output_file = args[2]

        if not output_file.endswith('.reaf'):
            output_file += '.reaf'
    
        convert_from_mp3(input_file, output_file)
    else:
        print("Invalid command")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> <args>")
        sys.exit(1)
    main(sys.argv[1:])
