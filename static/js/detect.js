// SignSpeak live detection:
// webcam/image -> MediaPipe Hands (browser) -> 21 landmarks -> POST /predict
// -> label + confidence -> on-screen text + speech + sentence builder.
(function () {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const img = document.getElementById('uploaded');
  const hint = document.getElementById('stage-hint');
  const predEl = document.getElementById('prediction');
  const confEl = document.getElementById('confidence');
  const confFill = document.getElementById('conf-fill');
  const sentenceEl = document.getElementById('sentence');
  const statusEl = document.getElementById('status');
  const btnCamera = document.getElementById('btn-camera');
  const btnSpeak = document.getElementById('btn-speak');
  const btnClear = document.getElementById('btn-clear');
  const btnSave = document.getElementById('btn-save');
  const fileInput = document.getElementById('file-input');
  const uploadWrap = document.getElementById('upload-wrap');
  const modeCamera = document.getElementById('mode-camera');
  const modeImage = document.getElementById('mode-image');

  let camera = null, running = false, mode = 'camera';
  let lastSent = 0, current = null;         // current = {label, confidence}
  let stableLabel = null, stableCount = 0, lastCommitted = null;
  let sentence = '';

  const hands = new Hands({
    locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
  });
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 0,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  hands.onResults(onResults);

  function speak(text) {
    if (!('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95;
    speechSynthesis.speak(u);
  }

  function setPrediction(label, confidence) {
    current = { label, confidence };
    predEl.textContent = label;
    confEl.textContent = 'Confidence: ' + Math.round(confidence * 100) + '%';
    confFill.style.width = Math.round(confidence * 100) + '%';
    btnSave.disabled = false;
  }

  function commit(label) {
    // A sign held steady counts once: append it to the message and speak it.
    if (label === lastCommitted) return;
    lastCommitted = label;
    sentence += (label.length === 1) ? label : (sentence && !sentence.endsWith(' ') ? ' ' : '') + label + ' ';
    sentenceEl.textContent = sentence || '…';
    btnSpeak.disabled = !sentence;
    speak(label.length === 1 ? label : label.charAt(0) + label.slice(1).toLowerCase());
  }

  async function predict(landmarks) {
    const now = Date.now();
    if (now - lastSent < 250) return;           // throttle to ~4 predictions/sec
    lastSent = now;
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: landmarks.map(p => ({ x: p.x, y: p.y })) }),
      });
      if (!res.ok) return;
      const data = await res.json();
      setPrediction(data.label, data.confidence);
      if (data.confidence >= 0.6) {
        if (data.label === stableLabel) stableCount += 1; else { stableLabel = data.label; stableCount = 1; }
        if (stableCount >= 3) commit(data.label);
      } else {
        stableCount = 0;
      }
    } catch (e) { /* network hiccup - keep showing the last prediction */ }
  }

  function onResults(results) {
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (mode === 'camera' && video.videoWidth) {
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }
    if (mode === 'image' && img.naturalWidth) {
      canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
    const lm = results.multiHandLandmarks && results.multiHandLandmarks[0];
    if (lm) {
      drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: '#FF6B5E', lineWidth: 3 });
      drawLandmarks(ctx, lm, { color: '#5B4BDB', lineWidth: 1, radius: 3 });
      hint.hidden = true;
      statusEl.textContent = 'Hand tracked: 21 landmarks';
      predict(lm);
    } else {
      statusEl.textContent = 'No hand detected — hold a sign inside the frame.';
      stableCount = 0; lastCommitted = null;
    }
    ctx.restore();
  }

  btnCamera.addEventListener('click', async () => {
    if (running) {
      camera.stop(); running = false;
      btnCamera.textContent = 'Start camera';
      statusEl.textContent = 'Camera stopped.';
      return;
    }
    camera = new Camera(video, {
      onFrame: async () => { await hands.send({ image: video }); },
      width: 640, height: 480,
    });
    await camera.start();
    running = true;
    hint.hidden = true;
    btnCamera.textContent = 'Stop camera';
    statusEl.textContent = 'Camera on — show a hand sign.';
  });

  function setMode(next) {
    mode = next;
    modeCamera.classList.toggle('active', mode === 'camera');
    modeImage.classList.toggle('active', mode === 'image');
    uploadWrap.hidden = mode !== 'image';
    btnCamera.disabled = mode !== 'camera';
    hint.hidden = mode === 'image';
    if (mode === 'image') hint.textContent = '';
  }
  modeCamera.addEventListener('click', () => setMode('camera'));
  modeImage.addEventListener('click', () => setMode('image'));

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;
    img.src = URL.createObjectURL(file);
    img.onload = async () => { await hands.send({ image: img }); };
  });

  btnSpeak.addEventListener('click', () => { if (sentence) speak(sentence); });
  btnClear.addEventListener('click', () => {
    sentence = ''; lastCommitted = null;
    sentenceEl.textContent = '…';
    btnSpeak.disabled = true;
  });
  btnSave.addEventListener('click', async () => {
    if (!current) return;
    const res = await fetch('/save_result', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(current),
    });
    statusEl.textContent = res.ok ? 'Saved to your Dashboard.' : 'Could not save the result.';
  });
})();
