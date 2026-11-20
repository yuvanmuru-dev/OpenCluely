const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { spawn } = require('child_process');
const logger = require('../core/logger').createServiceLogger('WHISPER-WORKER');

class WhisperWorkerService {
  constructor() {
    this.process = null;
    this.pythonPath = null;
    this.scriptPath = null;
    this.pending = new Map();
    this.sequence = 0;
    this.idleTimer = null;
    this.idleUnloadMs = 60000;
    this.readyInfo = null;
    this.useMlx = false;
  }

  configure({ pythonPath, scriptPath, idleUnloadMs = 60000, useMlx = false }) {
    const changed = this.pythonPath !== pythonPath || this.scriptPath !== scriptPath || this.useMlx !== useMlx;
    this.pythonPath = pythonPath;
    this.scriptPath = scriptPath;
    this.idleUnloadMs = idleUnloadMs;
    this.useMlx = useMlx;
    if (changed) {
      this.close();
    }
  }

  isConfigured() {
    return Boolean(
      this.pythonPath &&
      this.scriptPath &&
      fs.existsSync(this.pythonPath) &&
      fs.existsSync(this.scriptPath)
    );
  }

  async transcribe(audioPath, options = {}) {
    if (!this.isConfigured()) {
      throw new Error('Persistent Whisper worker is not configured');
    }
    this._clearIdleTimer();
    const result = await this._request({
      action: 'transcribe',
      audio_path: audioPath,
      model: options.model || 'small',
      language: options.language || 'auto',
      model_dir: options.modelDir || null,
      device: options.device || 'auto'
    });
    this._scheduleIdleUnload();
    return result;
  }

  async warmup(options = {}) {
    if (!this.isConfigured()) {
      throw new Error('Persistent Whisper worker is not configured');
    }
    this._clearIdleTimer();
    return this._request({
      action: 'warmup',
      model: options.model || 'small',
      model_dir: options.modelDir || null,
      device: options.device || 'auto'
    });
  }

  releaseWhenIdle() {
    if (this.process) {
      this._scheduleIdleUnload();
    }
  }

  _ensureProcess() {
    if (this.process && !this.process.killed) {
      return;
    }

    this.readyInfo = null;
    const args = ['-u', this.scriptPath];
    if (this.useMlx) args.push('--mlx');
    const child = spawn(this.pythonPath, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      env: this.useMlx ? { ...process.env, HF_HUB_DISABLE_IMPLICIT_TOKEN: '1', HF_HUB_DISABLE_PROGRESS_BARS: '1' } : process.env
    });
    this.process = child;

    const lines = readline.createInterface({ input: child.stdout });
    lines.on('line', (line) => {
      if (this.process === child) this._handleLine(line);
    });

    child.stderr.on('data', (chunk) => {
      const message = chunk.toString().trim();
      if (message) {
        logger.debug('Worker stderr', { message: message.substring(0, 1000) });
      }
    });

    child.on('error', (error) => {
      if (this.process !== child) return;
      logger.error('Whisper worker process error', { error: error.message });
      this._rejectAll(error);
    });

    child.on('close', (code) => {
      if (this.process !== child) return;
      const error = new Error(`Whisper worker exited with code ${code}`);
      if (code !== 0) logger.error(error.message);
      this.process = null;
      this.readyInfo = null;
      this._rejectAll(error);
    });
  }

  _handleLine(line) {
    let message;
    try {
      message = JSON.parse(line);
    } catch (error) {
      logger.warn('Ignoring non-JSON worker output', { line: line.substring(0, 500) });
      return;
    }

    if (message.event === 'ready') {
      this.readyInfo = message;
      logger.info('Persistent Whisper worker ready', message);
      return;
    }

    const pending = this.pending.get(message.id);
    if (!pending) {
      return;
    }
    this.pending.delete(message.id);
    clearTimeout(pending.timer);

    if (message.ok) {
      pending.resolve(message);
    } else {
      const error = new Error(message.error || 'Whisper worker request failed');
      error.workerTraceback = message.traceback;
      pending.reject(error);
    }
  }

  _request(payload, timeoutMs = 180000) {
    this._ensureProcess();
    const id = ++this.sequence;
    const request = { ...payload, id };

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Whisper worker timed out after ${timeoutMs}ms`));
        this.close();
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });

      this.process.stdin.write(`${JSON.stringify(request)}\n`, (error) => {
        if (error) {
          clearTimeout(timer);
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  _scheduleIdleUnload() {
    this._clearIdleTimer();
    this.idleTimer = setTimeout(() => {
      if (!this.process || this.pending.size > 0) {
        return;
      }
      this._request({ action: 'unload' }, 30000)
        .then(() => logger.info('Whisper model unloaded after idle timeout'))
        .catch((error) => logger.warn('Could not unload idle Whisper model', { error: error.message }));
    }, this.idleUnloadMs);
  }

  _clearIdleTimer() {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
  }

  _rejectAll(error) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  close() {
    this._clearIdleTimer();
    if (this.process && !this.process.killed) {
      try {
        this.process.kill();
      } catch (_) {
        // Ignore shutdown races.
      }
    }
    this.process = null;
    this.readyInfo = null;
    this._rejectAll(new Error('Whisper worker closed'));
  }
}

module.exports = WhisperWorkerService;
