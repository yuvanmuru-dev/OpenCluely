import gc
import json
import os
import sys
import traceback
import wave
from pathlib import Path

import numpy as np

USE_MLX = '--mlx' in sys.argv
SAMPLE_RATE = 16000

if USE_MLX:
    from huggingface_hub import snapshot_download
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder, transcribe as mlx_transcribe

    if not mx.metal.is_available():
        raise RuntimeError('MLX Whisper needs an available Apple GPU')
else:
    import torch
    import whisper


model = None
model_name = None
model_dir = None
model_device = None


def send(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def unload_model():
    global model, model_name, model_dir, model_device
    model = None
    model_name = None
    model_dir = None
    model_device = None
    gc.collect()
    if USE_MLX:
        ModelHolder.model = None
        ModelHolder.model_path = None
        mx.clear_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_mlx_model(request):
    global model_name, model_dir
    name = request.get('model') or 'base'
    if name == model_name and ModelHolder.model_path is not None:
        return ModelHolder.model_path

    unload_model()
    if Path(name).expanduser().is_dir():
        model_path = str(Path(name).expanduser().resolve())
    else:
        repos = {
            'tiny': 'mlx-community/whisper-tiny',
            'base': 'mlx-community/whisper-base-mlx',
            'small': 'mlx-community/whisper-small-mlx',
            'medium': 'mlx-community/whisper-medium-mlx',
            'large-v3': 'mlx-community/whisper-large-v3-mlx',
            'turbo': 'mlx-community/whisper-turbo',
        }
        if name not in repos:
            raise ValueError('Choose tiny, base, small, medium, large-v3, turbo, or a local MLX model folder')
        # Use cached weights for offline warmups. Only contact the Hub when the
        # required files are missing from the cache.
        files = ['config.json', 'weights.safetensors', 'weights.npz']
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:
            model_path = snapshot_download(repos[name], allow_patterns=files, local_files_only=True)
            cached = Path(model_path)
            if not (cached / 'config.json').is_file() or not any((cached / file).is_file() for file in files[1:]):
                raise LocalEntryNotFoundError('Model weights are missing from the cache')
        except LocalEntryNotFoundError:
            model_path = snapshot_download(repos[name], allow_patterns=files)

    ModelHolder.get_model(model_path, mx.float16)
    model_name = name
    model_dir = model_path
    return model_path


def ensure_model(request):
    global model, model_name, model_dir, model_device
    requested_name = request.get('model') or 'small'
    requested_dir = request.get('model_dir') or None
    requested_device = request.get('device') or 'auto'
    if requested_device == 'auto':
        requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but this Python environment has no CUDA-enabled PyTorch build'
        )

    if (
        model is None
        or model_name != requested_name
        or model_dir != requested_dir
        or model_device != requested_device
    ):
        unload_model()
        model = whisper.load_model(
            requested_name,
            device=requested_device,
            download_root=requested_dir,
        )
        model_name = requested_name
        model_dir = requested_dir
        model_device = requested_device

    return model, requested_device


def load_audio_input(audio_path):
    """Read the app's PCM WAV directly so ffmpeg is not required for recording."""
    if not str(audio_path).lower().endswith('.wav'):
        return audio_path

    try:
        with wave.open(audio_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            return audio_path

        samples = np.frombuffer(frames, dtype='<i2').astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        samples /= 32768.0

        if sample_rate != SAMPLE_RATE and samples.size:
            output_size = max(1, round(samples.size * SAMPLE_RATE / sample_rate))
            source_positions = np.arange(samples.size, dtype=np.float64)
            target_positions = np.linspace(0, samples.size - 1, output_size)
            samples = np.interp(target_positions, source_positions, samples).astype(np.float32)

        return samples
    except (OSError, EOFError, wave.Error, ValueError):
        # Non-PCM or malformed WAV files can still use Whisper's normal
        # filename loader when ffmpeg is available on the host.
        return audio_path


def transcribe(request):
    language = request.get('language') or None
    if language in ('auto', 'detect'):
        language = None

    audio = load_audio_input(request['audio_path'])
    options = dict(
        language=language,
        task='transcribe',
        verbose=None,
        temperature=0,
        condition_on_previous_text=False,
    )
    if USE_MLX:
        result = mlx_transcribe(audio, path_or_hf_repo=ensure_mlx_model(request), **options)
        device = 'metal'
    else:
        loaded_model, device = ensure_model(request)
        result = loaded_model.transcribe(audio, fp16=device == 'cuda', **options)

    return {
        'text': (result.get('text') or '').strip(),
        'language': result.get('language'),
        'device': device,
        'model': model_name,
        **device_info(),
    }


def device_info():
    if USE_MLX:
        return {'mlx': mx.__version__, 'cuda_available': False, 'gpu': mx.device_info().get('device_name')}
    cuda_available = torch.cuda.is_available()
    return {
        'torch': torch.__version__,
        'cuda_available': cuda_available,
        'torch_cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if cuda_available else None,
    }


send({
    'event': 'ready',
    'pid': os.getpid(),
    **device_info(),
})

for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue

    request_id = None
    try:
        request = json.loads(raw_line)
        request_id = request.get('id')
        action = request.get('action')

        if action == 'transcribe':
            send({'id': request_id, 'ok': True, **transcribe(request)})
        elif action == 'warmup':
            if USE_MLX:
                ensure_mlx_model(request)
                device = 'metal'
            else:
                _, device = ensure_model(request)
            send({
                'id': request_id,
                'ok': True,
                'warmed': True,
                'device': device,
                'model': model_name,
                'model_path': model_dir,
                **device_info(),
            })
        elif action == 'unload':
            unload_model()
            send({'id': request_id, 'ok': True, 'unloaded': True})
        elif action == 'shutdown':
            unload_model()
            send({'id': request_id, 'ok': True, 'shutdown': True})
            break
        else:
            raise ValueError(f'Unsupported worker action: {action}')
    except Exception as error:
        send({
            'id': request_id,
            'ok': False,
            'error': str(error),
            'traceback': traceback.format_exc(),
        })
