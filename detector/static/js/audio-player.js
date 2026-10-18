/**
 * Shared audio player module: spectrogram colormap, rendering, player controls.
 * Used by both index.html (events) and settings.html (calibration recordings).
 */

/* ── Spectrogram colormap ── */
const SPEC_CMAP = new Uint8Array(256 * 3);
(function() {
  const stops = [
    [0.00,   0,   0,   4],
    [0.15,  10,   8,  80],
    [0.30,  50,  10, 130],
    [0.45, 120,  28, 110],
    [0.60, 180,  50,  58],
    [0.75, 230, 130,  20],
    [0.90, 250, 210,  50],
    [1.00, 252, 255, 164],
  ];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let j = 0;
    while (j < stops.length - 2 && stops[j + 1][0] < t) j++;
    const [t0, r0, g0, b0] = stops[j];
    const [t1, r1, g1, b1] = stops[j + 1];
    const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    SPEC_CMAP[i * 3]     = Math.round(r0 + (r1 - r0) * f);
    SPEC_CMAP[i * 3 + 1] = Math.round(g0 + (g1 - g0) * f);
    SPEC_CMAP[i * 3 + 2] = Math.round(b0 + (b1 - b0) * f);
  }
})();

/**
 * Load spectrogram data from server and render onto canvas.
 * @param {HTMLElement} player - .clip-player element
 * @param {string} [spectrogramUrl] - override URL (default: player.dataset.clipUrl + '/spectrogram')
 */
function loadSpectrogram(player, spectrogramUrl) {
  const url = spectrogramUrl || player.dataset.clipUrl;
  if (!url) return;
  const specUrl = spectrogramUrl || (url + '/spectrogram');
  const canvas = player.querySelector('.spec-canvas');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, canvas.width || 300, canvas.height || 60);
  ctx.fillStyle = '#333';
  ctx.font = '12px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('loading...', (canvas.width || 300) / 2, 35);

  fetch(specUrl)
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(spec => {
      const raw = atob(spec.data);
      const arr = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);

      const off = document.createElement('canvas');
      off.width = spec.width;
      off.height = spec.height;
      const octx = off.getContext('2d');
      const img = octx.createImageData(spec.width, spec.height);
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        img.data[i * 4]     = SPEC_CMAP[v * 3];
        img.data[i * 4 + 1] = SPEC_CMAP[v * 3 + 1];
        img.data[i * 4 + 2] = SPEC_CMAP[v * 3 + 2];
        img.data[i * 4 + 3] = 255;
      }
      octx.putImageData(img, 0, 0);

      player._specImage = off;
      player._specDuration = spec.duration;
      drawSpecFrame(canvas, off, 0, spec.duration);
    })
    .catch(() => {
      ctx.fillStyle = '#1a1a1a';
      ctx.fillRect(0, 0, canvas.width || 300, canvas.height || 60);
      ctx.fillStyle = '#444';
      ctx.font = '12px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('no spectrogram', (canvas.width || 300) / 2, 35);
    });
}

/**
 * Draw spectrogram frame with optional overlays and progress line.
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLCanvasElement} offscreen - pre-rendered spectrogram image
 * @param {number} currentTime
 * @param {number} duration
 * @param {Object} [opts]
 * @param {Array} [opts.segments] - [{start, end, label}]
 * @param {Object} [opts.segColors] - {label: {fill, stroke}}
 * @param {number} [opts.dragStart] - drag selection start time
 * @param {number} [opts.dragEnd] - drag selection end time
 */
function drawSpecFrame(canvas, offscreen, currentTime, duration, opts) {
  const w = canvas.clientWidth || 300;
  const h = canvas.clientHeight || 60;
  if (canvas.width !== w * devicePixelRatio || canvas.height !== h * devicePixelRatio) {
    canvas.width = w * devicePixelRatio;
    canvas.height = h * devicePixelRatio;
  }
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);

  if (opts && opts.segments && duration > 0) {
    for (const seg of opts.segments) {
      const x1 = Math.round(seg.start / duration * canvas.width);
      const x2 = Math.round(seg.end / duration * canvas.width);
      const c = (opts.segColors && opts.segColors[seg.label]) || { fill: 'rgba(136,136,136,0.25)', stroke: 'rgba(136,136,136,0.7)' };
      ctx.fillStyle = c.fill;
      ctx.fillRect(x1, 0, x2 - x1, canvas.height);
      ctx.strokeStyle = c.stroke;
      ctx.lineWidth = devicePixelRatio;
      ctx.strokeRect(x1, 0, x2 - x1, canvas.height);
    }
  }

  if (opts && opts.dragStart != null && opts.dragEnd != null && duration > 0) {
    const x1 = Math.round(opts.dragStart / duration * canvas.width);
    const x2 = Math.round(opts.dragEnd / duration * canvas.width);
    ctx.fillStyle = 'rgba(76,175,80,0.2)';
    ctx.fillRect(Math.min(x1,x2), 0, Math.abs(x2-x1), canvas.height);
    ctx.strokeStyle = 'rgba(76,175,80,0.8)';
    ctx.lineWidth = 2 * devicePixelRatio;
    ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
    ctx.strokeRect(Math.min(x1,x2), 0, Math.abs(x2-x1), canvas.height);
    ctx.setLineDash([]);
  }

  if (currentTime > 0 && duration > 0) {
    const xPos = Math.round(currentTime / duration * canvas.width);
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 2 * devicePixelRatio;
    ctx.beginPath();
    ctx.moveTo(xPos, 0);
    ctx.lineTo(xPos, canvas.height);
    ctx.stroke();
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fillRect(0, 0, xPos, canvas.height);
  }
}

/* ── Player controls ── */
function pcSkip(btn, dir) {
  const player = btn.closest('.clip-player');
  const audio = player.querySelector('audio');
  if (!audio || !audio.duration) return;
  const skip = player._skipSec || Math.max(1, Math.round(audio.duration / 3));
  audio.currentTime = Math.max(0, Math.min(audio.duration, audio.currentTime + dir * skip));
}

const SPEED_STEPS = [1, 1.5, 2, 0.5];
function pcSpeed(btn) {
  const player = btn.closest('.clip-player');
  const audio = player.querySelector('audio');
  if (!audio) return;
  const cur = audio.playbackRate;
  let idx = SPEED_STEPS.indexOf(cur);
  idx = (idx + 1) % SPEED_STEPS.length;
  audio.playbackRate = SPEED_STEPS[idx];
  btn.textContent = SPEED_STEPS[idx] + '\u00D7';
}

/* ── Spectrogram progress animation ── */
let _specRaf = 0;
function _tickSpec() {
  let anyPlaying = false;
  document.querySelectorAll('.clip-player').forEach(player => {
    const audio = player.querySelector('audio');
    if (!audio || audio.paused || !player._specImage) return;
    anyPlaying = true;
    const canvas = player.querySelector('.spec-canvas');
    const opts = player._drawOpts || {};
    drawSpecFrame(canvas, player._specImage, audio.currentTime, player._specDuration, opts);
  });
  if (anyPlaying) { _specRaf = requestAnimationFrame(_tickSpec); }
  else { _specRaf = 0; }
}
function _startSpecTick() { if (!_specRaf) _specRaf = requestAnimationFrame(_tickSpec); }

/**
 * Initialize basic clip player: spectrogram loading, audio events, seek on click.
 * @param {HTMLElement} player - .clip-player element
 * @param {Object} [opts]
 * @param {string} [opts.spectrogramUrl] - override spectrogram URL
 * @param {string} [opts.audioSrc] - override audio src (set on audio element)
 */
function initBasicPlayer(player, opts) {
  loadSpectrogram(player, opts && opts.spectrogramUrl);

  const canvas = player.querySelector('.spec-canvas');
  canvas.addEventListener('click', function(e) {
    const audio = player.querySelector('audio');
    if (!audio || !audio.duration) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    audio.currentTime = Math.max(0, Math.min(audio.duration, x * audio.duration));
    if (player._specImage) drawSpecFrame(canvas, player._specImage, audio.currentTime, player._specDuration, player._drawOpts);
  });

  const audio = player.querySelector('audio');
  if (!audio) return;

  if (opts && opts.audioSrc) audio.src = opts.audioSrc;

  audio.addEventListener('loadedmetadata', function() {
    const skip = Math.max(1, Math.round(this.duration / 3));
    const back = player.querySelector('.pc-skip-back');
    const fwd = player.querySelector('.pc-skip-fwd');
    if (back) back.textContent = '\u25C0 ' + skip + 's';
    if (fwd) fwd.textContent = skip + 's \u25B6';
    player._skipSec = skip;
  });

  let _currentAudioRef = null;
  audio.addEventListener('play', function() {
    // Stop other players
    document.querySelectorAll('.clip-player audio').forEach(a => {
      if (a !== this && !a.paused) { a.pause(); a.currentTime = 0; }
    });
    _startSpecTick();
  });
  audio.addEventListener('pause', function() {
    if (player._specImage) drawSpecFrame(player.querySelector('.spec-canvas'), player._specImage, this.currentTime, player._specDuration, player._drawOpts);
  });
  audio.addEventListener('ended', function() {
    if (player._specImage) drawSpecFrame(player.querySelector('.spec-canvas'), player._specImage, 0, player._specDuration, player._drawOpts);
  });
  audio.addEventListener('seeked', function() {
    if (player._specImage) drawSpecFrame(player.querySelector('.spec-canvas'), player._specImage, this.currentTime, player._specDuration, player._drawOpts);
  });
}
