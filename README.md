# PacketLens

PacketLens classifies network traffic categories from the first bytes of each
flow. It converts anonymized flow bytes into a 28x28 grayscale image and
trains a small CNN to predict categories such as `web`, `streaming`, `voip`,
`chat`, and `file_transfer`.

This is a traffic-classification research project, not a payload inspection
or RAG application. It is designed for public PCAP datasets such as ISCX
VPN-nonVPN and ISCX Tor-nonTor.

## Pipeline

```text
PCAP files
  -> group packets by 5-tuple
  -> remove Ethernet and mask IPs/ports
  -> keep or zero-pad the first 784 bytes
  -> reshape to 28x28 grayscale
  -> CNN prediction
```

IP addresses, MAC addresses, and transport ports are masked before feature
generation. Train/test splits should be made by capture file or session, not
randomly by flow, to avoid leakage.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset manifest

Create a CSV whose `pcap` paths point to capture files and whose `label`
contains the traffic category:

```csv
pcap,label,session
data/chat.pcap,chat,chat-capture-01
data/video.pcap,streaming,video-capture-01
```

Build an image dataset:

```bash
python -m packetlens.dataset build data/manifest.csv data/flows.npz
python -m packetlens.visualize data/flows.npz \
  --output reports/sample_grid.png \
  --matrix-output reports/matrices.txt \
  --category-output reports/category_patterns.png \
  --per-category 6
```

To open the colored Matplotlib diagram on your desktop, add `--show`:

```bash
python -m packetlens.visualize data/flows.npz --count 25 --show
```

The sample grid is a colored matrix view: each square is one normalized byte
map, and its title gives the sample index and traffic category. The
`matrices.txt` output lists every displayed matrix row by row, together with
its category and capture session. Each cell is the byte value divided by 255,
so `0.0000` is a zero byte and `1.0000` is `0xff`. Use `--count` to control
how many flows are displayed and listed.

The category report shows several colored flow images for every class followed
by that class's mean pattern. It also reports mean intensity and variation,
which makes visual differences between classifications easier to compare.

Train and evaluate the CNN:

```bash
python -m packetlens.train data/flows.npz \
  --model mobilenet_v3_small --output models/mobilenet.pt
python -m packetlens.evaluate data/flows.npz \
  --checkpoint models/mobilenet.pt
```

The default model is an ImageNet-pretrained MobileNetV3-Small. Its first
convolution is adapted from three RGB channels to one grayscale channel by
averaging the pretrained filters, and the 28x28 map is resized internally
before the backbone. This gives you genuine transfer learning while retaining
the original low-resolution representation and its cell color intensities.

Use `--model native` to compare against the small CNN designed directly for
28x28 maps. Use `--model native_enhanced` for the normalized three-block CNN
with balanced sampling, label smoothing, AdamW, cosine learning-rate decay,
and validation-based best-checkpoint selection. The training split uses
stratified capture groups so every category is represented in validation and
test when enough capture groups are available. Use `--no-pretrained` for a
same-architecture ablation.

Every run writes inspectable artifacts to `reports/`:

- `training_history.json`
- `training_curves.png`
- `evaluation.json`
- `confusion_matrix.png`
- `predictions.csv`

Open the PNG files to see what the model actually learned and use the
confusion matrix to identify categories that are visually indistinguishable.

The enhanced USTC-TFC2016 benchmark is published in
[`reports/ustc-enhanced-stratified/`](./reports/ustc-enhanced-stratified/):

- [prediction diagram](./reports/ustc-enhanced-stratified/prediction-grid.png)
- [confusion matrix](./reports/ustc-enhanced-stratified/confusion_matrix.png)
- [evaluation report](./reports/ustc-enhanced-stratified/evaluation.json)
- [predictions](./reports/ustc-enhanced-stratified/predictions.csv)

This benchmark achieved 89.77% accuracy and 89.73% macro-F1 on 440 held-out
flows. The raw dataset and checkpoint remain local because they are large
generated artifacts; rerun the commands above to create them.

## Baseline and live classification

Compare against packet-flow statistics:

```bash
python -m packetlens.baseline data/flows.npz
```

Capture a live flow with Scapy and classify it:

```bash
sudo python -m packetlens.live --interface eth0 --checkpoint models/cnn.pt
```

Only capture traffic you are authorized to inspect. The live command prints
the predicted category and confidence; it does not persist raw packets.

## Research checks

The dataset builder supports the planned header/payload ablation:

- header bytes versus payload bytes
- configurable square byte windows
- masked preprocessing (the default)
- CNN versus flow-statistics baseline

For example, create a header-only dataset with `--headers-only`. The
anonymization step is always enabled so IP and port identity does not leak
into the classifier.

Keep capture sessions separated between train and test. Report macro-F1 in
addition to accuracy, especially for imbalanced categories.

## Complexity

The default CNN performs roughly 4.3M multiply-accumulate operations per
flow. Flow grouping is O(P) in packet count and preprocessing is O(N), where
N is the configured byte window.
