require('dotenv').config();

const speechService = require('../src/services/speech.service');

async function main() {
  const status = speechService.getStatus();

  console.log('Speech provider:', status.provider);
  console.log('Initialized:', status.isInitialized);
  console.log('Available:', speechService.isAvailable());
  console.log('Effective settings:', JSON.stringify(status.effectiveSettings, null, 2));

  const file = process.argv[2];
  if (file) {
    const start = performance.now();
    const first = await speechService.recognizeFromFile(file);
    const warmed = performance.now();
    const second = await speechService.recognizeFromFile(file);
    if (!first.trim() || !second.trim()) throw new Error('Expected transcribed speech in the audio file');
    console.log('Transcript:', second);
    console.log('First and repeat transcription (ms):', Math.round(warmed - start), Math.round(performance.now() - warmed));
    return;
  }

  const connection = await speechService.testConnection();
  console.log('Connection test:', JSON.stringify(connection, null, 2));
  if (!connection.success) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
}).finally(() => speechService.shutdown());
