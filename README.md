# Dimensionality Reduction — PCA, ICA & Autoencoders on MNIST

## Project Summary

Built and compared three dimensionality reduction algorithms — Principal Component Analysis (PCA), Independent Component Analysis (ICA), and Autoencoders — on the MNIST handwritten digit dataset. This project explores how to compress high dimensional data while retaining the information that matters.

**Dataset:** [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)

**Business Problem:** Can we compress 784-pixel digit images into a fraction of their original size — reducing infrastructure costs and speeding up predictions — without meaningfully hurting accuracy?

---

## What is Dimensionality Reduction?

Every ML model works with features. More features = more data to store, more compute to train, more memory to serve predictions. But not all features carry equal information.

Dimensionality reduction finds a compact representation of your data that preserves what matters and discards what doesn't:

```
784 pixels per image
→ Most corner pixels are always black → zero information
→ Adjacent pixels are highly correlated → redundant information
→ PCA finds 154 directions that capture 95% of all variation
→ 630 features discarded, 95% of signal retained
```

This is not just a technical optimisation. It is a product and business decision.

---

## Dataset Overview

```
Training images: 60,000
Test images:     10,000
Image size:      28 × 28 pixels = 784 features per image
Classes:         10 digits (0-9)
Pixel values:    0 (black) to 255 (white), normalised to 0-1
```

---

## Algorithm 1 — PCA (Principal Component Analysis)

### How It Works

PCA finds the directions in your data where variation is highest — called principal components — and projects everything onto those directions.

```
Step 1 — Find the direction where data varies most (Component 1)
Step 2 — Find the next direction, perpendicular to Component 1 (Component 2)
Step 3 — Keep finding directions until you have enough components
Step 4 — Project all data onto these directions
```

Each component is a combination of all original pixels weighted by importance. The first component might capture overall brightness, the second horizontal vs vertical strokes, and so on.

**It is linear** — can only find straight line relationships between features.

### Key Results

```
Components for 95% variance: 154  (from 784 original)
Components for 99% variance: 331
Compression ratio:           5.1x fewer features
```

### Accuracy vs Speed

| Model | Features | Accuracy | Training Time |
|---|---|---|---|
| Full Features | 784 | 90.50% | 9.06s |
| PCA (95%) | 154 | 90.66% | 1.35s |

**PCA was faster AND slightly more accurate.** The 630 dropped features were mostly noise — random pixel variations that confused the model. Removing them made classification easier.

### When To Use PCA

✅ General purpose dimensionality reduction
✅ When speed and interpretability matter
✅ Small to medium datasets
✅ When you need a quick, reliable baseline
✅ Real time inference with low latency requirements
❌ When data has complex non-linear patterns
❌ When maximum compression is needed
❌ When features are truly independent signals

---

## Algorithm 2 — ICA (Independent Component Analysis)

### How It Works

ICA was invented to solve the **cocktail party problem**:

```
3 people talking simultaneously in a room
3 microphones recording a mix of all 3 voices

Microphone 1 → 60% person A + 30% person B + 10% person C
Microphone 2 → 20% person A + 50% person B + 30% person C

ICA → separates the 3 original independent voices
```

Applied to MNIST, ICA finds the independent stroke patterns that combine to create every digit — curves, vertical lines, loops, diagonals.

**PCA asks:** "What directions have the most variance?"
**ICA asks:** "What independent signals are mixed together in this data?"

### Key Results

```
Components used: 50
Accuracy:        83.9%
```

ICA scored lower than PCA for two reasons:
- Fewer components (50 vs 154) — not a fair comparison
- ICA is designed for signal separation, not classification

### When To Use ICA

✅ Separating mixed signals (audio, financial, biological)
✅ EEG and brain signal analysis — separating individual brain region activations
✅ Removing artifacts from medical scans
✅ Financial risk factor separation — finding independent market forces
✅ Audio source separation
❌ General classification tasks — PCA is better
❌ Large datasets — computationally expensive
❌ When components need to be interpretable rankings

---

## Algorithm 3 — Autoencoders

### How It Works

An autoencoder is a neural network with two parts:

```
ENCODER                         DECODER
784 pixels                      784 pixels
    ↓                               ↑
  256 neurons    →→→→→→→→→→    256 neurons
    ↓                               ↑
  128 neurons    →→→→→→→→→→    128 neurons
    ↓                               ↑
   32 neurons ←── BOTTLENECK ──→ 32 neurons

The network learns to compress AND reconstruct
```

**Encoder** — compresses 784 pixels to 32 numbers
**Bottleneck** — the compressed representation
**Decoder** — reconstructs 784 pixels from 32 numbers

The network trains by minimising reconstruction error — how different the output is from the input. No labels needed.

**It is non-linear** — can find complex curved relationships that PCA completely misses.

### Key Results

```
Components: 32 (vs PCA's 154)
Accuracy:   85.0%
Compression: 24.5x (vs PCA's 5.1x)
```

Autoencoder achieves 85% accuracy with just 32 numbers representing each image — 24x compression. With more components it would likely outperform PCA because it captures non-linear patterns.

### When To Use Autoencoders

✅ Maximum compression needed
✅ Complex non-linear data (images, audio, text)
✅ Anomaly detection — reconstruction error flags unusual inputs
✅ Denoising — compress noisy data, reconstruct clean version
✅ Large datasets where non-linear patterns exist
✅ When GPU is available for training
❌ Small datasets — neural networks need lots of data
❌ When interpretability is required
❌ When training time is constrained
❌ Simple tabular data — PCA is faster and equally good

---

## Final Comparison

| Method | Components | Accuracy | Compression | Training Time |
|---|---|---|---|---|
| Full Features | 784 | 90.50% | 1x | 9.06s |
| PCA | 154 | 90.66% | 5.1x | 1.35s |
| Autoencoder | 32 | 85.00% | 24.5x | ~2 min |
| ICA | 50 | 83.90% | 15.7x | ~1 min |

**Best accuracy:** PCA — retains 95% variance, removes noise
**Best compression:** Autoencoder — 24.5x with 85% accuracy
**Best signal separation:** ICA — finds independent patterns
**Best overall for production:** PCA — fast, reliable, interpretable

---

## The Business Case For Dimensionality Reduction

This is where the real PM work begins. The numbers above are not just technical metrics — they translate directly into business outcomes.

### Infrastructure Cost Impact

Every feature you store and process has a cost:

```
Without PCA:
60,000 images × 784 features = 47,040,000 numbers stored
Training time: 9.06 seconds per model iteration

With PCA:
60,000 images × 154 features = 9,240,000 numbers stored
Training time: 1.35 seconds per model iteration

Cost reduction: 80% less storage, 85% faster training
```

At scale this compounds dramatically:

```
Processing 1,000,000 images daily:
Without PCA: 9.06s × 1M = 104 days of compute
With PCA:    1.35s × 1M = 15 days of compute

Annual cloud compute savings: significant
```

For a PM managing an ML infrastructure budget — dimensionality reduction is one of the highest ROI technical decisions you can make.

### Training Time Impact On Product Velocity

Faster training = faster iteration = faster product improvement:

```
Without PCA: 9 seconds per training run
With PCA:    1.35 seconds per training run

If you retrain daily on new data:
Without PCA: model updates take hours
With PCA:    model updates take minutes
```

This directly impacts how quickly your team can respond to:
- Model performance degradation
- New data distributions
- A/B test results
- User feedback

A PM who understands this can make the case for PCA investment in terms the business understands — shipping speed, not algorithm theory.

### Customer Experience Impact

Dimensionality reduction affects the end user in two ways:

**Prediction speed (latency):**
```
Without PCA: model processes 784 features per prediction
With PCA:    model processes 154 features per prediction

Result: faster API responses, better user experience
Critical for: real time applications, mobile apps, high traffic products
```

**Model reliability:**
```
Full features accuracy: 90.50%
PCA accuracy:           90.66%
```

Counterintuitively, PCA improved accuracy by removing noisy features. In production, noisy features cause inconsistent predictions — the model behaves differently on similar inputs. Removing noise makes the model more reliable and predictable for users.

**Autoencoder anomaly detection:**
Autoencoders have a unique production capability — anomaly detection. When an unusual input arrives (blurry image, corrupted scan, out of distribution digit), the autoencoder struggles to reconstruct it and produces high reconstruction error. This can be used to:
- Flag low confidence predictions for human review
- Detect data quality issues in incoming data
- Identify fraudulent or manipulated inputs

### Model Reliability

High dimensional models are fragile in production:
- More features = more ways for the model to find spurious correlations
- Spurious correlations break when data distribution shifts
- Users experience inconsistent, unpredictable model behaviour

Dimensionality reduction forces the model to rely on genuine signal:
```
PCA keeps: the 154 directions that explain 95% of real variation
PCA drops: the 630 directions that explain noise and artifacts

Result: model relies on real patterns, not noise
        more consistent predictions across different input conditions
        more robust to data quality issues in production
```

---

## The AI PM Role In Dimensionality Reduction

Dimensionality reduction decisions are not made by data scientists alone. Every key decision requires product judgment:

### Decision 1 — How Much Compression Is Acceptable?

```
95% variance retained (154 components) → 90.66% accuracy
99% variance retained (331 components) → higher accuracy, less compression
80% variance retained (~50 components) → lower accuracy, more compression
```

This is a product tradeoff: **accuracy vs cost vs speed.**

The PM must define:
- What is the minimum acceptable accuracy for this use case?
- What is the maximum acceptable latency?
- What is the infrastructure budget?

No data scientist can answer these questions. They are product decisions.

### Decision 2 — Which Algorithm For Which Use Case?

| Use Case | PM Should Choose |
|---|---|
| Compress images for faster inference | PCA |
| Separate EEG signals for medical device | ICA |
| Detect fraudulent documents | Autoencoder (anomaly detection) |
| Reduce storage costs for tabular data | PCA |
| Find non-linear patterns in user behaviour | Autoencoder |

The PM needs enough understanding to ask the right questions — not to implement the algorithm.

### Decision 3 — When To Retrain

Dimensionality reduction models degrade over time:
- New handwriting styles not seen in training
- Camera quality changes affecting pixel distributions
- New digit formats in different regions

The PM must define:
- Retraining frequency (weekly, monthly, on drift detection?)
- Performance thresholds that trigger retraining
- Monitoring dashboards to catch degradation early

### Decision 4 — Build vs Buy

Before building a custom autoencoder:
- Pre-trained image encoders (ResNet, EfficientNet) compress images better than custom autoencoders
- For text, transformer encoders (BERT) outperform custom solutions
- Building custom is justified only when domain is highly specialised

A PM who knows this prevents weeks of unnecessary engineering work.

---

## Key PM Takeaways

**1. Dimensionality reduction is a cost decision as much as a technical one.**
Every feature stored and processed has a dollar cost. PCA reduced our compute by 85%. That is not a technical win — it is a business win.

**2. More features is not always better.**
Our PCA model with 154 features outperformed the full 784 feature model. Noise reduction through compression is a real phenomenon with real accuracy benefits.

**3. The compression target is a product decision.**
95% vs 99% variance retention is not a data science call. It depends on your accuracy requirements, latency budget, and infrastructure costs. The PM owns this decision.

**4. Autoencoders enable anomaly detection for free.**
Any product that processes user-generated content (images, documents, audio) can use reconstruction error as a quality signal — flagging unusual or corrupted inputs before they reach the model.

**5. Dimensionality reduction directly impacts user experience.**
Faster predictions, more reliable outputs, better anomaly handling — all of these translate into measurable user satisfaction improvements.

---

## Tools & Libraries

```python
datasets                    # Hugging Face dataset loading
numpy                       # Numerical operations
scikit-learn                # PCA, ICA, evaluation metrics
sklearn.decomposition.PCA
sklearn.decomposition.FastICA
torch                       # PyTorch for Autoencoder
torch.nn                    # Neural network layers
matplotlib                  # Visualisation
seaborn                     # Statistical visualisation
Pillow                      # Image processing
python-dotenv               # Environment variable management
huggingface_hub             # Authentication
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/shambhavichaugule/dimensionality-reduction.git
cd dimensionality-reduction

# Activate virtual environment
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token to .env
echo "HF_TOKEN=your_token_here" > .env

# Run
python dimensionality_reduction.py
```

---



---
