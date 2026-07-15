/**
 * voice.js – Voice input utility for MedTrack
 * Provides toggleVoiceInput() used in symptom_checker.html and other forms.
 */

const _voiceRecognitions = {};

function toggleVoiceInput(fieldId) {
  const field = document.getElementById(fieldId);
  const statusEl = document.getElementById('voice-status-' + fieldId);

  // Browser support check
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    if (statusEl) {
      statusEl.textContent = '⚠️ Voice input not supported in this browser.';
      statusEl.classList.remove('hidden');
      setTimeout(() => statusEl.classList.add('hidden'), 3000);
    }
    return;
  }

  // If already recording for this field, stop it
  if (_voiceRecognitions[fieldId] && _voiceRecognitions[fieldId]._running) {
    _voiceRecognitions[fieldId].stop();
    return;
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition._running = false;
  _voiceRecognitions[fieldId] = recognition;

  recognition.onstart = function () {
    recognition._running = true;
    if (statusEl) {
      statusEl.textContent = 'Listening... Speak now 🎙️';
      statusEl.classList.remove('hidden');
    }
  };

  recognition.onresult = function (event) {
    const transcript = event.results[0][0].transcript;
    if (field) {
      field.value = field.value ? field.value + ' ' + transcript : transcript;
    }
  };

  recognition.onerror = function (event) {
    console.warn('Speech recognition error:', event.error);
    if (statusEl) {
      statusEl.textContent = '⚠️ Voice error: ' + event.error;
      statusEl.classList.remove('hidden');
    }
  };

  recognition.onend = function () {
    recognition._running = false;
    if (statusEl) {
      setTimeout(() => statusEl.classList.add('hidden'), 1500);
    }
  };

  recognition.start();
}
