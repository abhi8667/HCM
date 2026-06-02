document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictor-form');
  const select = document.getElementById('sequence-select');
  const customInput = document.getElementById('custom-sequence');
  const customPos = document.getElementById('custom-position');
  const resultPanel = document.getElementById('prediction-result');
  const statusEl = document.getElementById('result-status');
  const confEl = document.getElementById('result-confidence');
  const seqTrack = document.getElementById('sequence-track');
  const infoPanel = document.getElementById('info-panel');
  const molViewerEl = document.getElementById('mol-viewer');
  const loadingOverlay = document.getElementById('loading-overlay');

  const baseSequence = "VLKSQRVKATX"; // default fallback

  // Populate dropdown
  if (typeof variantData !== 'undefined') {
    variantData.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.variant;
      opt.textContent = `${item.variant} (Residue ${item.position})`;
      select.appendChild(opt);
    });
    // Set first valid as selected just to have a default visually
    select.selectedIndex = 1;
  }

  // Clear custom input when select changes
  select.addEventListener('change', () => {
    customInput.value = '';
    customPos.value = '';
  });

  // Initial Render (if there's data)
  let initialVar = (typeof variantData !== 'undefined' && variantData.length > 0) ? variantData[0] : null;
  let currentSeq = initialVar ? initialVar.sequence : baseSequence;
  renderSequence(currentSeq);
  init3DViewer(initialVar ? initialVar.position : 100);

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    
    let isPathogenic = false;
    let scoreText = '';
    let sequenceToRender = '';
    let posToHighlight = 100;

    const customSeq = customInput.value.toUpperCase();
    
    if (customSeq && customSeq.length === 11) {
      // Use Live Backend for Custom Sequences
      customInput.value = customSeq;
      posToHighlight = customPos.value ? parseInt(customPos.value, 10) : 100;
      
      // Show loading state
      resultPanel.style.display = 'block';
      resultPanel.style.borderLeft = '6px solid #999';
      statusEl.textContent = 'Analyzing...';
      statusEl.style.color = '#999';
      statusEl.style.webkitTextFillColor = '#999';
      confEl.textContent = 'Running Two-Tower Neural Network...';
      
      fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: customSeq, position: posToHighlight })
      })
      .then(res => res.json())
      .then(result => {
        if (result.error) {
          confEl.textContent = result.error;
          return;
        }
        const pathogenic = result.is_pathogenic;
        resultPanel.classList.remove('prediction-anim');
        void resultPanel.offsetWidth;
        resultPanel.classList.add('prediction-anim');
        resultPanel.style.borderLeft = `6px solid ${pathogenic ? '#ff3333' : '#2ea043'}`;
        statusEl.textContent = result.prediction;
        statusEl.style.color = pathogenic ? '#ff3333' : '#2ea043';
        statusEl.style.webkitTextFillColor = pathogenic ? '#ff3333' : '#2ea043';
        confEl.textContent = (result.calibrated_score * 100).toFixed(1) + '% (' + result.model + ')';
        
        renderSequence(customSeq);
        update3DViewer(posToHighlight, pathogenic);
      })
      .catch(err => {
        confEl.textContent = 'Backend offline. Start: python website/backend/app.py';
        statusEl.textContent = 'Error';
        statusEl.style.color = '#ff3333';
        statusEl.style.webkitTextFillColor = '#ff3333';
      });
      return; // async, don't continue below
    } else {
      // Use Dropdown Variant Logic
      if (!select.value) return;
      const selectedVar = variantData.find(v => v.variant === select.value);
      if (!selectedVar) return;

      isPathogenic = selectedVar.score > 0.5;
      scoreText = (selectedVar.score * 100).toFixed(1) + '%';
      sequenceToRender = selectedVar.sequence;
      posToHighlight = selectedVar.position;
    }

    // Animate prediction panel
    resultPanel.style.display = 'block';
    resultPanel.classList.remove('prediction-anim');
    void resultPanel.offsetWidth; // trigger reflow
    resultPanel.classList.add('prediction-anim');

    resultPanel.style.borderLeft = `6px solid ${isPathogenic ? '#ff3333' : '#2ea043'}`;
    statusEl.textContent = isPathogenic ? 'Pathogenic' : 'Benign';
    statusEl.style.color = isPathogenic ? '#ff3333' : '#2ea043';
    statusEl.style.webkitTextFillColor = isPathogenic ? '#ff3333' : '#2ea043';
    confEl.textContent = scoreText;

    // Re-render sequence
    renderSequence(sequenceToRender);
    
    // Update 3D Viewer position
    update3DViewer(posToHighlight, isPathogenic);
  });

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

  function update3DViewer(pos, isPathogenic, isInit=false) {
    if (!window.currentViewer) return;
    const viewer = window.currentViewer;
    
    // Reset all styles to a clean, minimal white cartoon
    viewer.setStyle({}, { cartoon: { color: 'white', opacity: 0.5 } });
    viewer.removeAllLabels();

    let mainColor = isPathogenic ? '#ff3333' : '#2ea043';
    if (isInit) mainColor = '#666';

    // Highlight local neighborhood (Blast Radius within 5 Angstroms)
    viewer.setStyle({within: {distance: 5, sel: {resi: pos}}}, { 
      stick: {color: 'lightgray', radius: 0.2},
      cartoon: {color: 'lightgray', opacity: 0.8}
    });

    // Emphasize the exact mutation site so it's not overridden by the neighborhood style
    viewer.addStyle({resi: pos}, { sphere: {color: mainColor, radius: 1.5} });

    // Add main label
    viewer.addLabel(isInit ? "Variant Location" : (isPathogenic ? "Pathogenic Disruption" : "Benign Change"), { 
      font: 'Inter', fontSize: 12, showBackground: true, 
      backgroundColor: isInit ? "rgba(100, 100, 100, 0.8)" : (isPathogenic ? "rgba(255, 51, 51, 0.8)" : "rgba(46, 160, 67, 0.8)"), 
      position: {resi: pos}, 
      fontColor: "white", inFront: true 
    });
    
    viewer.zoomTo({resi: pos}, 1000); // smooth zoom over 1s
    viewer.render();
  }

  function init3DViewer(initialPos) {
    if (typeof $3Dmol === 'undefined') {
      console.error("3Dmol.js not loaded");
      return;
    }
    
    const viewer = $3Dmol.createViewer(molViewerEl, { backgroundColor: '#f5f5f5' });
    window.currentViewer = viewer;
    
    // Fetch AlphaFold model for EXACT 1:1 mapping with P08590 Uniprot sequence
    fetch("model.pdb")
      .then(response => {
        if (!response.ok) throw new Error("Network response was not ok");
        return response.text();
      })
      .then(data => {
        viewer.addModel(data, "pdb");
        
        // Setup hover interactions
        viewer.setHoverable({}, true, 
          function(atom, viewer) {
            if (!atom.label) {
              viewer.spin(false); // Auto-pause on hover
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
              viewer.spin(true, 0.15); // Resume spinning
            }
          }
        );

        update3DViewer(initialPos, false, true);
        viewer.spin(true, 0.15); 
        loadingOverlay.style.display = 'none';
      })
      .catch(error => {
        console.error("Error loading 3D model:", error);
        loadingOverlay.textContent = "Error loading 3D model. Please try again.";
      });
  }

  // Scroll Reveal Observer for "Emerging" effect
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
