document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('predictor-form');
  const select = document.getElementById('sequence-select');
  const resultPanel = document.getElementById('prediction-result');
  const statusEl = document.getElementById('result-status');
  const confBadge = document.getElementById('result-confidence-badge');
  const calibratedEl = document.getElementById('result-calibrated-score');
  const rawEl = document.getElementById('result-raw-score');
  const rfEl = document.getElementById('result-rf-score');
  const esmStatusEl = document.getElementById('result-esm-status');
  const explanationsList = document.getElementById('explanations-list');
  
  const seqTrack = document.getElementById('sequence-track');
  const infoPanel = document.getElementById('info-panel');
  const molViewerEl = document.getElementById('mol-viewer');
  const loadingOverlay = document.getElementById('loading-overlay');
  
  // Loader & Error Panel Elements
  const loaderOverlay = document.getElementById('predict-loader');
  const loaderText = document.getElementById('predict-loader-text');
  const errorPanel = document.getElementById('predict-error');
  const errorMessage = document.getElementById('predict-error-message');
  const errorRetryBtn = document.getElementById('predict-error-retry');
  
  // Tab Elements
  const tabPreset = document.getElementById('tab-preset');
  const tabCustom = document.getElementById('tab-custom');
  const presetContent = document.getElementById('preset-content');
  const customContent = document.getElementById('custom-content');
  
  // Custom Form Inputs
  const customGene = document.getElementById('custom-gene');
  const customPos = document.getElementById('custom-position');
  const customRef = document.getElementById('custom-ref');
  const customAlt = document.getElementById('custom-alt');
  const customSeq = document.getElementById('custom-sequence');
  
  // Collapsible Advanced Trigger
  const advToggle = document.getElementById('advanced-toggle-btn');
  const advContainer = document.getElementById('advanced-input-container');

  const baseSequence = "VLKSQRVKATX"; // default fallback
  const apiBaseUrl = 'http://localhost:5000';
  let activeTab = 'preset';
  let currentVariants = [];

  // Initialize Tabs Toggling
  tabPreset.addEventListener('click', () => {
    tabPreset.classList.add('active');
    tabCustom.classList.remove('active');
    presetContent.style.display = 'block';
    customContent.style.display = 'none';
    activeTab = 'preset';
  });

  tabCustom.addEventListener('click', () => {
    tabCustom.classList.add('active');
    tabPreset.classList.remove('active');
    presetContent.style.display = 'none';
    customContent.style.display = 'block';
    activeTab = 'custom';
  });

  // Collapsible Advanced Options Panel
  advToggle.addEventListener('click', () => {
    if (advContainer.style.display === 'none') {
      advContainer.style.display = 'block';
      advToggle.textContent = 'Advanced Sequence Options ▴';
    } else {
      advContainer.style.display = 'none';
      advToggle.textContent = 'Advanced Sequence Options ▾';
    }
  });

  // Populate presets dropdown
  function populatePresets(variants) {
    select.innerHTML = '<option value="">Select a preset variant...</option>';
    variants.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.variant;
      opt.textContent = `${item.variant} (Residue ${item.position})`;
      select.appendChild(opt);
    });
    // Set default selection
    if (variants.length > 0) {
      select.selectedIndex = 1;
    }
  }

  // Load variant data dynamically from backend
  function loadPresets() {
    errorPanel.style.display = 'none';
    fetch(`${apiBaseUrl}/variants`)
      .then(res => {
        if (!res.ok) throw new Error("HTTP error " + res.status);
        return res.json();
      })
      .then(data => {
        if (data.variants && data.variants.length > 0) {
          currentVariants = data.variants;
          populatePresets(currentVariants);
          console.log("Presets loaded from backend API.");
        } else {
          useFallbackPresets();
        }
      })
      .catch(err => {
        console.warn("Backend offline or unreachable. Falling back to local static variantData:", err);
        useFallbackPresets();
      });
  }

  function useFallbackPresets() {
    if (typeof variantData !== 'undefined') {
      currentVariants = variantData;
      populatePresets(currentVariants);
    } else {
      console.error("Local variantData is undefined.");
    }
  }

  // Initial Load
  loadPresets();

  let initialVar = (typeof variantData !== 'undefined' && variantData.length > 0) ? variantData[0] : null;
  let currentSeq = initialVar ? initialVar.sequence : baseSequence;
  renderSequence(currentSeq);
  init3DViewer(initialVar ? initialVar.position : 100);

  // Form Submission handler
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    errorPanel.style.display = 'none';
    resultPanel.style.display = 'none';

    let payload = {};

    if (activeTab === 'preset') {
      // ─── Preset Variant Logic ───
      if (!select.value) {
        showError("Please select a variant from the dropdown list.");
        return;
      }
      const selectedVar = currentVariants.find(v => v.variant === select.value);
      if (!selectedVar) {
        showError("Selected variant details not found.");
        return;
      }
      
      // Presets in local data.js are MYL3 variants
      payload = {
        gene: "MYL3",
        position: parseInt(selectedVar.position, 10),
        ref_aa: selectedVar.ref_aa,
        alt_aa: selectedVar.alt_aa,
        sequence_window: selectedVar.sequence
      };
    } else {
      // ─── Custom Variant Logic ───
      const gene = customGene.value;
      const position = customPos.value;
      const ref = customRef.value;
      const alt = customAlt.value;
      const windowSeq = customSeq.value.trim().toUpperCase();

      // Client-Side Validation
      if (!gene) {
        showError("Please select a target gene.");
        return;
      }
      if (!position || isNaN(position) || parseInt(position, 10) <= 0) {
        showError("Please provide a valid positive integer residue position.");
        return;
      }
      if (!ref) {
        showError("Please select the wildtype (reference) residue.");
        return;
      }
      if (!alt) {
        showError("Please select the mutant (alternative) residue.");
        return;
      }
      if (ref === alt) {
        showError("The reference and mutant residues cannot be the same amino acid.");
        return;
      }
      if (windowSeq && windowSeq.length !== 11) {
        showError("The sequence window must be exactly 11 characters if specified.");
        return;
      }

      payload = {
        gene: gene,
        position: parseInt(position, 10),
        ref_aa: ref,
        alt_aa: alt,
        sequence_window: windowSeq
      };
    }

    // Trigger API inference
    runInference(payload);
  });

  // Run Inference Function
  function runInference(payload) {
    // Show Loading Panel
    loaderOverlay.style.display = 'flex';
    loaderText.textContent = "Connecting to Two-Tower prediction service...";
    
    // Dynamic loader steps
    const loaderSteps = [
      "Establishing pipeline context...",
      "Resolving wildtype sequence window...",
      "Querying ESM-2 model (this may take 1-3 seconds on first run)...",
      "Computing biophysical properties...",
      "Synthesizing two-tower late-fusion embeddings...",
      "Calculating calibrator scores & explainability profiles..."
    ];
    let stepIdx = 0;
    const interval = setInterval(() => {
      if (loaderOverlay.style.display === 'flex' && stepIdx < loaderSteps.length) {
        loaderText.textContent = loaderSteps[stepIdx++];
      }
    }, 800);

    fetch(`${apiBaseUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(res => {
      clearInterval(interval);
      if (!res.ok) {
        return res.json().then(jsonErr => {
          throw new Error(jsonErr.messages ? jsonErr.messages.join(" ") : jsonErr.error || "Inference server error");
        });
      }
      return res.json();
    })
    .then(result => {
      loaderOverlay.style.display = 'none';
      
      // Update UI results panel
      const pathogenic = result.is_pathogenic;
      resultPanel.classList.remove('prediction-anim');
      void resultPanel.offsetWidth; // trigger reflow
      resultPanel.classList.add('prediction-anim');
      resultPanel.style.display = 'block';

      // Border Color based on classification
      resultPanel.style.borderLeft = `6px solid ${pathogenic ? '#ff3333' : '#2ea043'}`;
      statusEl.textContent = result.prediction;
      statusEl.style.color = pathogenic ? '#ff3333' : '#2ea043';
      statusEl.style.webkitTextFillColor = pathogenic ? '#ff3333' : '#2ea043';

      // Score Value
      calibratedEl.textContent = (result.calibrated_score * 100).toFixed(1) + '%';
      rawEl.textContent = (result.raw_score * 100).toFixed(1) + '%';
      rfEl.textContent = (result.rf_score * 100).toFixed(1) + '%';
      esmStatusEl.textContent = result.esm_status;

      // Confidence badge styling
      confBadge.textContent = `${result.confidence} Confidence`;
      confBadge.className = 'badge'; // reset
      confBadge.classList.add(result.confidence.toLowerCase());

      // Version badge
      document.getElementById('result-version-badge').textContent = result.model_version || 'v1.0';

      // Feature Explanations rendering
      explanationsList.innerHTML = '';
      if (result.explanations && result.explanations.length > 0) {
        result.explanations.forEach(exp => {
          const item = document.createElement('div');
          item.className = 'explanation-item';
          
          const meta = document.createElement('div');
          meta.className = 'explanation-meta';
          
          const nameSpan = document.createElement('span');
          nameSpan.innerHTML = `<strong>${formatFeatureName(exp.feature)}</strong> (value: <span class="explanation-val">${exp.value}</span>)`;
          
          const impSpan = document.createElement('span');
          impSpan.className = 'explanation-val';
          impSpan.textContent = `${(exp.importance * 100).toFixed(1)}%`;
          
          meta.appendChild(nameSpan);
          meta.appendChild(impSpan);
          
          const barBg = document.createElement('div');
          barBg.className = 'explanation-bar-bg';
          
          const barFill = document.createElement('div');
          barFill.className = 'explanation-bar-fill';
          barFill.style.width = '0%';
          
          barBg.appendChild(barFill);
          item.appendChild(meta);
          item.appendChild(barBg);
          explanationsList.appendChild(item);

          // Animate widths
          setTimeout(() => {
            barFill.style.width = `${exp.importance * 100}%`;
          }, 100);
        });
      } else {
        explanationsList.innerHTML = '<p style="font-size: 13px; color: var(--air-text-muted);">No explainability metadata returned by server.</p>';
      }

      // Render the sequence window returned by the server
      renderSequence(result.sequence_window);
      
      // Update 3D Molecular structure viewer
      // Only highlight 3D if the gene is MYL3 (as the loaded model.pdb structure is for MYL3)
      if (result.gene === 'MYL3') {
        update3DViewer(result.position, pathogenic);
      } else {
        // Fallback or message about structure mismatch
        loggerInfoStructure(`Note: Molecular structure structure viewer shows MYL3 (UniProt: P08590). 3D highlighting is disabled for other genes (${result.gene}).`);
      }
    })
    .catch(err => {
      clearInterval(interval);
      loaderOverlay.style.display = 'none';
      showError(err.message || "Failed to contact the backend service. Make sure backend is running: python app.py");
    });
  }

  function loggerInfoStructure(msg) {
    console.log(msg);
  }

  function showError(msg) {
    errorPanel.style.display = 'flex';
    errorMessage.textContent = msg;
  }

  // Setup error retry button click
  errorRetryBtn.addEventListener('click', () => {
    errorPanel.style.display = 'none';
    loadPresets();
  });

  // Feature name formatter for UI display
  function formatFeatureName(name) {
    if (name === "ESM-2 Delta Embedding") return "ESM-2 Neural Context Delta";
    if (name === "grantham_score") return "Grantham Chemical Distance";
    if (name === "rel_position") return "Relative Chain Location";
    if (name === "position") return "Chain Residue position";
    if (name === "protein_length") return "Protein Chain Length";
    if (name === "size_change") return "Amino Acid Volume Change";
    if (name === "charge_change") return "Amino Acid Charge Change";
    if (name === "in_domain") return "Domain Structural Status";
    if (name === "in_helix") return "Helix Secondary Structure";
    if (name === "in_strand") return "Beta Strand Secondary Structure";
    if (name === "in_turn") return "Beta Turn Secondary Structure";
    if (name === "in_secondary") return "Secondary Structure Annotation";
    if (name === "in_disordered") return "Disordered Region Status";
    if (name === "in_coiled") return "Coiled Coil Annotation";
    if (name === "in_functional_site") return "Functional Site Annotation";
    if (name === "in_ptm_site") return "PTM Active Site status";
    if (name.startsWith("win_")) {
      const idx = name.replace("win_", "").replace("_size", " Volume").replace("_charge", " Charge");
      return `Window position ${idx}`;
    }
    return name.replace(/_/g, ' ');
  }

  // 1D sequence window renderer
  function renderSequence(seq) {
    seqTrack.innerHTML = '';
    seq.split('').forEach((char, index) => {
      const pos = index - 5;
      const isMutated = (pos === 0); 
      const size = Math.round(char.charCodeAt(0) * 1.5);
      const charge = ['R','K','H'].includes(char) ? 1 : ['D','E'].includes(char) ? -1 : 0;

      const block = document.createElement('div');
      block.className = `aa-block ${isMutated ? 'mutated' : ''}`;
      
      const posEl = document.createElement('div');
      posEl.className = 'aa-pos';
      posEl.textContent = pos > 0 ? `+${pos}` : pos;

      const card = document.createElement('div');
      card.className = 'aa-card';
      
      const front = document.createElement('div');
      front.className = 'aa-front';
      front.textContent = char;
      
      card.appendChild(front);
      block.appendChild(posEl);
      block.appendChild(card);

      block.addEventListener('mouseenter', () => showInfo(pos, char, isMutated, size, charge));
      block.addEventListener('mouseleave', hideInfo);

      seqTrack.appendChild(block);
    });
  }

  function showInfo(pos, char, isMutated, size, charge) {
    infoPanel.classList.remove('placeholder');
    infoPanel.innerHTML = `
      <div class="info-grid">
        <div class="info-col">
          <span class="info-label">Relative Pos</span>
          <span class="info-value">${pos}</span>
        </div>
        <div class="info-col">
          <span class="info-label">Amino Acid</span>
          <span class="info-value ${isMutated ? 'alt' : ''}">${char}</span>
        </div>
        <div class="info-col">
          <span class="info-label">Volume</span>
          <span class="info-value">${size} Å³</span>
        </div>
        <div class="info-col">
          <span class="info-label">Charge</span>
          <span class="info-value">${charge > 0 ? '+'+charge : charge}</span>
        </div>
      </div>
    `;
  }

  function hideInfo() {
    infoPanel.classList.add('placeholder');
    infoPanel.innerHTML = '<span class="metal-text">Hover over a residue to view structural properties</span>';
  }

  // 3D molecular structure updates
  function update3DViewer(pos, isPathogenic, isInit=false) {
    if (!window.currentViewer) return;
    const viewer = window.currentViewer;
    
    // Reset all styles
    viewer.setStyle({}, { cartoon: { color: 'white', opacity: 0.5 } });
    viewer.removeAllLabels();

    let mainColor = isPathogenic ? '#ff3333' : '#2ea043';
    if (isInit) mainColor = '#666';

    // Highlight neighborhood residues
    viewer.setStyle({within: {distance: 5, sel: {resi: pos}}}, { 
      stick: {color: 'lightgray', radius: 0.2},
      cartoon: {color: 'lightgray', opacity: 0.8}
    });

    // Emphasize exact site
    viewer.addStyle({resi: pos}, { sphere: {color: mainColor, radius: 1.5} });

    // Annotation Label
    viewer.addLabel(isInit ? "Variant Location" : (isPathogenic ? "Pathogenic Disruption" : "Benign Change"), { 
      font: 'Inter', fontSize: 12, showBackground: true, 
      backgroundColor: isInit ? "rgba(100, 100, 100, 0.8)" : (isPathogenic ? "rgba(255, 51, 51, 0.8)" : "rgba(46, 160, 67, 0.8)"), 
      position: {resi: pos}, 
      fontColor: "white", inFront: true 
    });
    
    viewer.zoomTo({resi: pos}, 1000);
    viewer.render();
  }

  // Initialize 3D molecular viewer
  function init3DViewer(initialPos) {
    if (typeof $3Dmol === 'undefined') {
      console.error("3Dmol.js not loaded");
      return;
    }
    
    const viewer = $3Dmol.createViewer(molViewerEl, { backgroundColor: '#f5f5f5' });
    window.currentViewer = viewer;
    
    // Load local PDB model (represents MYL3 structure)
    fetch("model.pdb")
      .then(response => {
        if (!response.ok) throw new Error("PDB file fetch failure");
        return response.text();
      })
      .then(data => {
        viewer.addModel(data, "pdb");
        
        viewer.setHoverable({}, true, 
          function(atom, viewer) {
            if (!atom.label) {
              viewer.spin(false);
              atom.label = viewer.addLabel(atom.resn + " " + atom.resi, {
                position: atom,
                backgroundColor: "rgba(0,0,0,0.8)",
                fontColor: "white",
                fontSize: 12,
                inFront: true
              });
            }
          },
          function(atom, viewer) {
            if (atom.label) {
              viewer.removeLabel(atom.label);
              delete atom.label;
              viewer.spin(true, 0.15);
            }
          }
        );

        update3DViewer(initialPos, false, true);
        viewer.spin(true, 0.15); 
        loadingOverlay.style.display = 'none';
      })
      .catch(error => {
        console.error("Error loading 3D molecular structure pdb:", error);
        loadingOverlay.textContent = "Error loading 3D structural model.";
      });
  }

  // Reveal Animations
  const revealElements = document.querySelectorAll('.reveal');
  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
      }
    });
  }, {
    root: null,
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px"
  });

  revealElements.forEach(el => revealObserver.observe(el));
});
