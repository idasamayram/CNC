% Vibration-based Fault Classification with Multi-domain Explainability
% [Your Name] — Supervisors: Prof. Dr. Grégoire Montavon; (HHI) Maximilian Dreyer | Institution: Fraunhofer HHI
% [Today’s Date]

# Motivation and Problem
- Industrial need: accurate AND explainable fault detection for maintenance decisions
- Deep models achieve ~99.8% accuracy but are black boxes for engineers
- Goal: trustworthy, actionable AI for vibration analysis

# Research Questions
- RQ1: Accurate and interpretable 1D-CNN for multi-axial vibration
- RQ2: Extend attributions to frequency/time–frequency (virtual inspection layers)
- RQ3: Which domain yields the most faithful explanations?
- RQ4: What physically meaningful patterns does the model learn?

# Dataset and Labels
- Bosch CNC machining dataset: 3 machines (M01–M03), 15 operations, 2 kHz, tri-axial
- Labels: OK (normal) vs NOK (faulty) by expert operators
- Real-world challenges: noise, class imbalance, cross-machine variability

# Preprocessing Pipeline
- Trim OK samples; 5 s windows; NOK: 50% overlap
- Downsample to 400 Hz (2000 points) for efficiency
- Z-score normalization per axis and operation using OK-only stats
- StratifiedGroupKFold with grouping to prevent leakage

# Models Evaluated
- Classical baselines: SVM, Random Forest, Gradient Boosting (60 engineered features)
- Neural baselines: MLP, TCN, 1D-CNN-Freq
- Proposed: 1D-CNN-GN vs 1D-CNN-Wide (selected)

# Proposed 1D-CNN-Wide (ours)
- 4 conv blocks, LeakyReLU, no normalization layers, MaxPool strategy
- Progressive channels (16→32→64→128), Kaiming init
- Receptive field ≈ 592 steps (1.48 s @ 400 Hz)
- Explainability-friendly design: clean gradient/relevance pathways

# Classification Results
- 1D-CNN-Wide: Accuracy/F1 ≈ 0.998 (best overall)
- TCN and 1D-CNN-GN comparable; MLP underperforms
- 1D-CNN-Freq weaker than time-domain CNN

# Generalization to Novel Data
- Unseen machine/regime: Acc 0.959; Balanced Acc 0.812
- Distribution shifts across axes explain performance drop
- Implication: need domain adaptation/augmentation

# Multi-Domain Explainability
- Virtual inspection layers: insert T and T^{-1} identity loop
- Attributions in time, frequency (DFT), time–frequency (STDFT)
- Methods: LRP, Grad×Input, SmoothGrad, Occlusion (relevance conservation for LRP)

# Qualitative Attributions (Summary)
- Normal: relevance at very low frequencies (0–10 Hz), X-axis dominant
- Faulty: strong relevance ~150 Hz, higher magnitudes, Z-axis dominant
- STDFT: persistent bands for faults; localized patterns for normals

# Faithfulness Evaluation
- Window flipping with class-specific references:
  - Normal: sensitive to noise, robust to zeroing
  - Faulty: sensitive to zeroing, robust to mild noise
- Time-domain AUC ratios (Least/Most): LRP/Occlusion ~1.9–1.98
- Frequency-domain ratios: ~1.25–1.27 across methods

# Feature-/Concept-Level Insights
- Normal: Low-Frequency Structured — low spectral centroid/peak (0–10 Hz), higher peak-to-average, X-axis
- Faulty: High-Frequency Irregular — ~150 Hz, higher magnitude variability/energy, Z-axis
- Bridge to engineering knowledge of vibration analysis

# Limitations
- Domain shift sensitivity (unseen machine/regime)
- Compute overhead for frequency/time–frequency attributions
- Hyperparameter sensitivity (LRP rules, STFT windows)
- No ground-truth attributions (reliance on indirect evaluation)

# Future Work
- Domain adaptation; richer augmentations
- Attention mechanisms for focus and inherent interpretability
- Wavelet-based virtual layers; optimized FFT-LRP; on-demand explanations
- Concept bottlenecks; expert-defined frequency bands; actionable dashboards

# Conclusion
- 1D-CNN-Wide + multi-domain XAI bridges model decisions with engineering insight
- Transparent, high-utility diagnostics for vibration monitoring
- Practical path toward trustworthy AI-assisted maintenance