# How to Enable SAM3 Segmentation

## Step 1: Install SAM (Segment Anything Model)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Step 2: Install PyTorch (if not already installed)

For CUDA (GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch torchvision
```

## Step 3: Download SAM Checkpoint

Download one of these checkpoints (choose based on your needs):

### Option A: ViT-H (Largest, Best Quality) - Recommended
```bash
# Download to project root
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Option B: ViT-L (Medium)
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

### Option C: ViT-B (Smallest, Fastest)
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**Windows PowerShell:**
```powershell
# ViT-H (default)
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -OutFile "sam_vit_h_4b8939.pth"

# Or ViT-B (faster, smaller)
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "sam_vit_b_01ec64.pth"
```

## Step 4: Place Checkpoint File

Place the downloaded `.pth` file in your project root directory:
```
crowd_management/
├── sam_vit_h_4b8939.pth  ← Place here
├── backend/
├── live_crowd_display.py
└── ...
```

## Step 5: Update Config (if using different model)

If you downloaded ViT-B or ViT-L, update `backend/config.py`:

```python
SAM_CHECKPOINT: str = "sam_vit_b_01ec64.pth"  # For ViT-B
SAM_MODEL_TYPE: str = "vit_b"  # vit_h, vit_l, or vit_b
```

## Step 6: Run with SAM3

### For Live Display:
```bash
python live_crowd_display.py
```
It will automatically detect and enable SAM if checkpoint exists.

### For Backend API:
The backend will auto-detect SAM if checkpoint is found.

## Verification

When SAM3 is enabled, you'll see:
- Green segmentation masks over detected people
- Contour outlines around each person
- Stats showing: `"detector": "YOLOv8 + SAM"`

## Troubleshooting

1. **"SAM checkpoint not found"**: Make sure the `.pth` file is in the project root
2. **"CUDA not available"**: SAM will use CPU (slower but works)
3. **Import errors**: Make sure you installed SAM: `pip install git+https://github.com/facebookresearch/segment-anything.git`

