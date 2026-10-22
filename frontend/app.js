/**
 * FraudGuard AI - Frontend Application Logic
 * Manages live prediction scoring, preset loading, UI state, and interactive visual gallery.
 */

document.addEventListener('DOMContentLoaded', () => {
  // State variables
  let cachedSamples = { fraud_samples: [], legit_samples: [] };
  let currentFraudIdx = 0;
  let currentLegitIdx = 0;

  // DOM Elements
  const predictionForm = document.getElementById('prediction-form');
  const modelSelector = document.getElementById('model-selector');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdValDisplay = document.getElementById('threshold-val');
  const presetThDefault = document.getElementById('preset-th-default');
  const presetThOpt = document.getElementById('preset-th-opt');
  const btnLoadLegit = document.getElementById('btn-load-legit');
  const btnLoadFraud = document.getElementById('btn-load-fraud');
  const btnReset = document.getElementById('btn-reset-form');
  const otherFeaturesContainer = document.getElementById('other-features-container');

  // Results DOM Elements
  const gaugeRing = document.getElementById('gauge-ring');
  const fraudPctDisplay = document.getElementById('fraud-pct-display');
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictRule = document.getElementById('verdict-rule');
  const riskPill = document.getElementById('risk-pill');
  const modelTagBadge = document.getElementById('model-tag-badge');
  const contributionsList = document.getElementById('contributions-list');

  // Gallery & Modal DOM Elements
  const galleryTabs = document.querySelectorAll('.gallery-tab');
  const galleryItems = document.querySelectorAll('.gallery-item');
  const modalBackdrop = document.getElementById('img-modal');
  const modalImg = document.getElementById('modal-img');
  const modalCaption = document.getElementById('modal-caption');
  const modalCloseBtn = document.getElementById('modal-close-btn');

  // 1. Generate full inputs for all remaining PCA features (V1 to V28)
  const topFeatures = ['V17', 'V14', 'V12', 'V10', 'V11', 'V4'];
  for (let i = 1; i <= 28; i++) {
    const featName = `V${i}`;
    if (!topFeatures.includes(featName)) {
      const wrap = document.createElement('div');
      wrap.className = 'mini-feat-wrap';
      wrap.innerHTML = `
        <label for="input-${featName}">${featName}</label>
        <input type="number" id="input-${featName}" step="0.01" value="0.00" class="mini-input" data-feat="${featName}">
      `;
      otherFeaturesContainer.appendChild(wrap);
    }
  }

  // 2. Synchronize sliders and numeric inputs for top features
  topFeatures.forEach(feat => {
    const slider = document.getElementById(`slider-${feat}`);
    const numInput = document.getElementById(`input-${feat}`);

    if (slider && numInput) {
      slider.addEventListener('input', (e) => {
        numInput.value = parseFloat(e.target.value).toFixed(2);
        schedulePredict();
      });

      numInput.addEventListener('input', (e) => {
        slider.value = e.target.value;
        schedulePredict();
      });
    }
  });

  // 3. Threshold Slider & Presets
  thresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    thresholdValDisplay.textContent = val;
    updateThresholdChips(val);
    schedulePredict();
  });

  presetThDefault.addEventListener('click', () => {
    thresholdSlider.value = 0.50;
    thresholdValDisplay.textContent = '0.50';
    updateThresholdChips('0.50');
    schedulePredict();
  });

  presetThOpt.addEventListener('click', () => {
    thresholdSlider.value = 0.95;
    thresholdValDisplay.textContent = '0.95';
    updateThresholdChips('0.95');
    schedulePredict();
  });

  function updateThresholdChips(val) {
    presetThDefault.classList.toggle('active', Math.abs(val - 0.50) < 0.01);
    presetThOpt.classList.toggle('active', Math.abs(val - 0.95) < 0.01);
  }

  modelSelector.addEventListener('change', () => {
    const labels = {
      'scratch_lr': 'Scratch LR',
      'sk_lr': 'Sklearn LR',
      'rf': 'Random Forest'
    };
    modelTagBadge.textContent = labels[modelSelector.value] || 'Model';
    executePrediction();
  });

  // 4. Fetch Sample Test Transactions from Backend
  fetch('/api/samples')
    .then(res => res.json())
    .then(data => {
      cachedSamples = data;
    })
    .catch(err => {
      console.warn('Could not load samples from server, using built-in defaults:', err);
    });

  // 5. Preset Button Handlers
  btnLoadLegit.addEventListener('click', () => {
    if (cachedSamples.legit_samples && cachedSamples.legit_samples.length > 0) {
      const sample = cachedSamples.legit_samples[currentLegitIdx % cachedSamples.legit_samples.length];
      currentLegitIdx++;
      populateForm(sample);
    } else {
      populateWithFallbackLegit();
    }
    executePrediction();
  });

  btnLoadFraud.addEventListener('click', () => {
    if (cachedSamples.fraud_samples && cachedSamples.fraud_samples.length > 0) {
      const sample = cachedSamples.fraud_samples[currentFraudIdx % cachedSamples.fraud_samples.length];
      currentFraudIdx++;
      populateForm(sample);
    } else {
      populateWithFallbackFraud();
    }
    executePrediction();
  });

  btnReset.addEventListener('click', () => {
    document.getElementById('input-amount').value = '25.00';
    document.getElementById('input-time').value = '45000';
    for (let i = 1; i <= 28; i++) {
      const el = document.getElementById(`input-V${i}`);
      const sl = document.getElementById(`slider-V${i}`);
      if (el) el.value = '0.00';
      if (sl) sl.value = '0';
    }
    executePrediction();
  });

  function populateForm(sample) {
    if (sample.Amount !== undefined) document.getElementById('input-amount').value = sample.Amount.toFixed(2);
    if (sample.Time !== undefined) document.getElementById('input-time').value = Math.round(sample.Time);

    for (let i = 1; i <= 28; i++) {
      const key = `V${i}`;
      if (sample[key] !== undefined) {
        const val = Number(sample[key]).toFixed(2);
        const inputEl = document.getElementById(`input-${key}`);
        const sliderEl = document.getElementById(`slider-${key}`);
        if (inputEl) inputEl.value = val;
        if (sliderEl) sliderEl.value = val;
      }
    }
  }

  function populateWithFallbackFraud() {
    document.getElementById('input-amount').value = '9.99';
    document.getElementById('input-time').value = '84200';
    const fraudVals = { V17: -4.8, V14: -5.6, V12: -4.2, V10: -3.8, V11: 3.5, V4: 4.1 };
    for (const [k, v] of Object.entries(fraudVals)) {
      const inEl = document.getElementById(`input-${k}`);
      const slEl = document.getElementById(`slider-${k}`);
      if (inEl) inEl.value = v;
      if (slEl) slEl.value = v;
    }
  }

  function populateWithFallbackLegit() {
    document.getElementById('input-amount').value = '42.50';
    document.getElementById('input-time').value = '38000';
    const legitVals = { V17: 0.1, V14: 0.2, V12: 0.05, V10: -0.05, V11: -0.1, V4: 0.05 };
    for (const [k, v] of Object.entries(legitVals)) {
      const inEl = document.getElementById(`input-${k}`);
      const slEl = document.getElementById(`slider-${k}`);
      if (inEl) inEl.value = v;
      if (slEl) slEl.value = v;
    }
  }

  // 6. Gather Form Data and Execute Prediction
  let debounceTimer = null;
  function schedulePredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(executePrediction, 150);
  }

  predictionForm.addEventListener('submit', (e) => {
    e.preventDefault();
    executePrediction();
  });

  function executePrediction() {
    const features = {
      'Time': parseFloat(document.getElementById('input-time').value) || 0.0,
      'Amount': parseFloat(document.getElementById('input-amount').value) || 0.0,
    };

    for (let i = 1; i <= 28; i++) {
      const el = document.getElementById(`input-V${i}`);
      features[`V${i}`] = el ? (parseFloat(el.value) || 0.0) : 0.0;
    }

    const payload = {
      model_type: modelSelector.value,
      threshold: parseFloat(thresholdSlider.value),
      features: features,
    };

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(res => res.json())
      .then(data => {
        renderPredictionResults(data);
      })
      .catch(err => {
        console.error('Prediction failed:', err);
      });
  }

  function renderPredictionResults(res) {
    const pPct = res.fraud_probability_percent;
    fraudPctDisplay.textContent = `${pPct.toFixed(1)}%`;

    // Dynamic Gauge Gradient Calculation
    const deg = Math.min(360, Math.max(0, (pPct / 100) * 360));
    if (res.is_fraud) {
      gaugeRing.style.background = `conic-gradient(#ef4444 0deg, #f97316 ${deg}deg, #1e293b ${deg}deg 360deg)`;
      gaugeRing.style.boxShadow = '0 0 30px rgba(239, 68, 68, 0.4)';
    } else {
      gaugeRing.style.background = `conic-gradient(#10b981 0deg, #06b6d4 ${deg}deg, #1e293b ${deg}deg 360deg)`;
      gaugeRing.style.boxShadow = '0 0 25px rgba(16, 185, 129, 0.25)';
    }

    // Verdict Banner
    if (res.is_fraud) {
      verdictBanner.className = 'verdict-banner fraud';
      verdictIcon.textContent = '⚠';
      verdictTitle.textContent = 'FRAUD ALERT DETECTED';
      riskPill.className = 'risk-pill pill-critical';
      riskPill.textContent = `${res.risk_level} RISK`;
    } else {
      verdictBanner.className = 'verdict-banner legit';
      verdictIcon.textContent = '✓';
      verdictTitle.textContent = 'TRANSACTION APPROVED';
      riskPill.className = 'risk-pill pill-low';
      riskPill.textContent = `${res.risk_level} RISK`;
    }

    verdictRule.textContent = res.decision_rule;

    // Feature Contribution List
    if (res.top_feature_contributions && res.top_feature_contributions.length > 0) {
      contributionsList.innerHTML = '';
      res.top_feature_contributions.forEach(item => {
        const row = document.createElement('div');
        row.className = 'contrib-item';
        const isHazard = item.impact_score > 0;
        row.innerHTML = `
          <span class="contrib-name">${item.feature} (val: ${item.raw_value.toFixed(2)})</span>
          <span class="contrib-score ${isHazard ? 'contrib-fraud' : 'contrib-legit'}">
            ${isHazard ? '+' : ''}${item.impact_score.toFixed(2)} (${item.direction})
          </span>
        `;
        contributionsList.appendChild(row);
      });
    }
  }

  // 7. Gallery Filter Tabs
  galleryTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      galleryTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const filter = tab.dataset.filter;
      galleryItems.forEach(item => {
        if (filter === 'all' || item.dataset.category === filter) {
          item.style.display = 'block';
        } else {
          item.style.display = 'none';
        }
      });
    });
  });

  // 8. Image Lightbox Modal
  document.querySelectorAll('.gallery-image-wrap').forEach(wrap => {
    wrap.addEventListener('click', () => {
      const img = wrap.querySelector('.gallery-img');
      const title = wrap.parentElement.querySelector('.fig-title')?.textContent || 'Figure';
      modalImg.src = img.src;
      modalCaption.textContent = title;
      modalBackdrop.classList.add('open');
      modalBackdrop.setAttribute('aria-hidden', 'false');
    });
  });

  modalCloseBtn.addEventListener('click', closeModal);
  modalBackdrop.addEventListener('click', (e) => {
    if (e.target === modalBackdrop) closeModal();
  });

  function closeModal() {
    modalBackdrop.classList.remove('open');
    modalBackdrop.setAttribute('aria-hidden', 'true');
  }

  // 9. Initial Prediction on load
  executePrediction();
});
