import os, time, queue, threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import platform # Import platform module

# Platform-specific imports for volume control
if platform.system() == "Windows":
    try:
        from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    except ImportError:
        print("pycaw library not found. Windows volume control will not be available. pip install pycaw")
        AudioUtilities = None
        ISimpleAudioVolume = None
elif platform.system() == "Linux":
    try:
        import pulsectl
    except ImportError:
        print("pulsectl library not found. Linux volume control will not be available. pip install pulsectl")
        pulsectl = None
        
else: # macOS or other unsupported
    AudioUtilities = None
    ISimpleAudioVolume = None
    pulsectl = None

from transformers import pipeline # Import Hugging Face pipeline

# Configuration
SAMPLE_RATE = 16000
DURATION = 2  # Seconds to record per chunk (increased from 1, adjust as needed)
SILENCE_THRESHOLD = 0.005 # RMS threshold for silence detection (increased from 0.001, adjust as needed)

# Trigger phrases
TRIGGER_PHRASES = ["звукабы", "тих тих тих", "шумно"]  # Trigger phrases in Russian
REVERSE_TRIGGER_PHRASES = ["неважно", "всё нормально", "не важно"]  # Reverse trigger phrases in Russian
TRIGGER_PHRASE_LANGUAGE = "russian"  # Language for Hugging Face Whisper (e.g., "english", "russian")

# Application volume control settings
# For Linux, these names should match 'application.name' or 'application.process.binary'
# from `pactl list sink-inputs`. Examples: "Spotify", "Firefox", "discord"
APPS_TO_LOWER_VOLUME = ["Discord.exe"]  # Adjust for Linux, e.g., "discord"
APPS_TO_MUTE = ["Spotify.exe"]          # Adjust for Linux, e.g., "spotify"
LOWERED_VOLUME_LEVEL = 0.3
MUTED_VOLUME_LEVEL = 0.0

# Hugging Face ASR Pipeline
HF_MODEL_NAME = "openai/whisper-tiny"
asr_pipeline = None # Will be loaded in main

# Audio processing queue
audio_queue = queue.Queue()

# Global state for volume management
original_volumes = {}
volumes_adjusted = False
pulse_client = None # Global PulseAudio client for Linux

# Helper class for Linux volume control to mimic pycaw's interface
class PulseAudioAppVolume:
    def __init__(self, pulse_client_ref, sink_input_info):
        self.pulse_client_ref = pulse_client_ref # Reference to the global pulse client
        self.sink_input_info = sink_input_info

    def GetMasterVolume(self):
        # Refresh sink_input_info to get current volume
        si = self.pulse_client_ref.sink_input_info(self.sink_input_info.index)
        # Return average volume of channels, or first channel if preferred
        # pulsectl volumes are typically 0.0 to 1.0 (or >1.0 for overamplification)
        return sum(si.volume.values) / len(si.volume.values) if si.volume.values else 0.0


    def SetMasterVolume(self, level, _=None): # Match pycaw signature
        self.pulse_client_ref.volume_set_all_chans(self.sink_input_info, level)

def audio_callback(indata, frames, time, status):
  """Callback function for the audio input stream."""
  if status:
    print(f"Error: {status}")
  audio_queue.put(indata.copy())

def process_audio():
  """Process audio chunks and detect the trigger phrase."""
  global volumes_adjusted, asr_pipeline 
  if not asr_pipeline:
    print("Error: Hugging Face ASR pipeline not loaded.")
    return

  while True:
    audio_data = []
    # Collect audio for DURATION seconds
    start_time = time.time()
    while time.time() - start_time < DURATION:
      if not audio_queue.empty():
        audio_data.append(audio_queue.get())
    
    if audio_data:
      # Convert to the format expected
      audio_np = np.concatenate(audio_data, axis=0).astype(np.float32)
      
      # Silence detection: Calculate RMS of the audio chunk
      rms = np.sqrt(np.mean(audio_np**2))
      print(f"RMS: {rms:.4f} (Threshold: {SILENCE_THRESHOLD})")
      if rms < SILENCE_THRESHOLD:
        # print(f"Silence detected (RMS: {rms:.4f}), skipping ASR.")
        continue # Skip ASR for this chunk

      # Save as a temporary WAV file
      temp_file = "temp_audio.wav"
      sf.write(temp_file, audio_np, SAMPLE_RATE)
      
      # Transcribe using Hugging Face ASR pipeline
      try:
        # Forcing the language during transcription
        # The language codes for Whisper via Hugging Face are typically full names like "english", "russian", "german"
        transcription_result = asr_pipeline(temp_file, generate_kwargs={"language": TRIGGER_PHRASE_LANGUAGE})
        text = transcription_result["text"]
        print(f"Heard: {text}")
          
        text_lower = text.lower()
        # Check for trigger phrases
        if any(phrase.lower() in text_lower for phrase in TRIGGER_PHRASES):
          print(f"Trigger phrase detected: {TRIGGER_PHRASES}")
          apply_volume_changes()
        elif any(phrase.lower() in text_lower for phrase in REVERSE_TRIGGER_PHRASES):
          print(f"Reverse trigger phrase detected: {REVERSE_TRIGGER_PHRASES}")
          revert_volume_changes()
            
      except Exception as e:
        print(f"Error with Hugging Face ASR pipeline: {e}") # Updated error message
      
      # Clean up
      try:
        os.remove(temp_file)
      except Exception as e:
        print(f"Error removing temp file: {e}") # More specific error on cleanup

def get_app_volume(app_name_pattern):
  """Get the audio session for a specific application."""
  global pulse_client
  if platform.system() == "Windows":
    if not AudioUtilities:
        return None
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume_interface = session._ctl.QueryInterface(ISimpleAudioVolume)
        if session.Process and session.Process.name() == app_name_pattern:
            return volume_interface
  elif platform.system() == "Linux":
    if not pulsectl or not pulse_client:
        return None
    try:
        sink_inputs = pulse_client.sink_input_list()
        for si in sink_inputs:
            app_name = si.proplist.get('application.name', '').lower()
            app_binary = si.proplist.get('application.process.binary', '').lower()
            # Match against common properties
            if app_name_pattern.lower() in app_name or app_name_pattern.lower() in app_binary:
                return PulseAudioAppVolume(pulse_client, si)
    except pulsectl.PulseOperationFailed as e:
        print(f"PulseAudio operation failed: {e}")
        return None
  return None

def apply_volume_changes():
  """Lowers volume for specified apps and mutes others, storing original volumes."""
  global original_volumes, volumes_adjusted

  if volumes_adjusted:
    print("Volumes already adjusted. Use reverse trigger phrase to revert.")
    return

  print("Applying volume changes...")
  current_originals = {}

  # Lower volume for specified apps
  for app_name in APPS_TO_LOWER_VOLUME:
    volume_control = get_app_volume(app_name)
    if volume_control:
      try:
        current_originals[app_name] = volume_control.GetMasterVolume()
        volume_control.SetMasterVolume(LOWERED_VOLUME_LEVEL, None)
        print(f"{app_name} volume set to {LOWERED_VOLUME_LEVEL*100}%")
      except Exception as e:
        print(f"Error adjusting volume for {app_name}: {e}")
    else:
      print(f"Could not find {app_name} to lower volume.")
      
  # Mute specified apps
  for app_name in APPS_TO_MUTE:
    volume_control = get_app_volume(app_name)
    if volume_control:
      try:
        current_originals[app_name] = volume_control.GetMasterVolume()
        volume_control.SetMasterVolume(MUTED_VOLUME_LEVEL, None)
        print(f"{app_name} muted.")
      except Exception as e:
        print(f"Error muting {app_name}: {e}")
    else:
      print(f"Could not find {app_name} to mute.")

  if current_originals: # Only update state if any volume was actually changed
    original_volumes = current_originals
    volumes_adjusted = True
    print("Original volumes stored.")
  else:
    print("No applications found to adjust volumes.")


def revert_volume_changes():
  """Reverts volumes to their original levels."""
  global original_volumes, volumes_adjusted

  if not volumes_adjusted:
    print("Volumes are not currently adjusted. No action taken.")
    return

  if not original_volumes:
    print("No original volumes stored to revert to. Resetting state.")
    volumes_adjusted = False 
    return

  print("Reverting volume changes...")
  for app_name, original_level in original_volumes.items():
    volume_control = get_app_volume(app_name)
    if volume_control:
      try:
        volume_control.SetMasterVolume(original_level, None)
        print(f"{app_name} volume restored to {original_level*100:.0f}%")
      except Exception as e:
        print(f"Error reverting volume for {app_name}: {e}")
    else:
      print(f"Could not find {app_name} to revert volume (was it closed?).")
  
  original_volumes.clear()
  volumes_adjusted = False
  print("Volumes reverted and original volume store cleared.")

if __name__ == "__main__":
  if platform.system() == "Linux" and pulsectl:
    try:
        pulse_client = pulsectl.Pulse('mutee-volume-control')
    except Exception as e:
        print(f"Failed to connect to PulseAudio: {e}")
        pulsectl = None # Disable pulsectl features if connection fails

  try:
    print(f"Loading Hugging Face ASR pipeline with model: {HF_MODEL_NAME}...")
    # For CPU explicitly: asr_pipeline = pipeline("automatic-speech-recognition", model=HF_MODEL_NAME, device=-1)
    # For GPU (if available and PyTorch with CUDA is installed):
    # import torch
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # asr_pipeline = pipeline("automatic-speech-recognition", model=HF_MODEL_NAME, device=device)
    # Simpler default (tries GPU if available, else CPU):
    asr_pipeline = pipeline("automatic-speech-recognition", model=HF_MODEL_NAME)
    print("Hugging Face ASR pipeline loaded successfully.")
  except Exception as e:
    print(f"Error loading Hugging Face ASR pipeline: {e}")
    print("Please ensure 'transformers', 'torch', and 'torchaudio' are installed correctly, and ffmpeg is available.")
    exit(1)

  print("Starting audio streaming to Hugging Face ASR...")
  print(f"Listening for trigger phrases: {TRIGGER_PHRASES}")
  print(f"Listening for reverse trigger phrases: {REVERSE_TRIGGER_PHRASES}")
  print(f"Apps to lower volume: {APPS_TO_LOWER_VOLUME} to {LOWERED_VOLUME_LEVEL*100}%")
  print(f"Apps to mute: {APPS_TO_MUTE}")
  
  # Start audio processing thread
  processing_thread = threading.Thread(target=process_audio, daemon=True)
  processing_thread.start()
  
  # Start audio input stream
  with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE):
    try:
      print("Press Ctrl+C to stop")
      while True:
        time.sleep(0.1)
    except KeyboardInterrupt:
      print("Stopping...")
    finally:
        if platform.system() == "Linux" and pulse_client:
            pulse_client.close()