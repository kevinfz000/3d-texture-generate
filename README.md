# 3d-texture-generate
AI learns to texture 3D meshes: feed OBJ, get MTL+TGA. 33-D geo features drive GAN generator &amp; MLP for color; CNN+attention+VGG loss yield 256² RGB. Train on triplets, batch-infer new assets.
• `results<number>` – outputs from experiment **t1**  
• `results-<number>` – outputs from experiment **t2**  
• `results-neo-<number>` – outputs from experiment **t3**  
• `sourcecode` – full code progression from **tbeta to t4**  
• `dev log` – chronological development notes  
• `report` – AI-generated project summary  
• `report1` – answers to:  
  1. Which GitHub repos were tried (with links)  
  2. Results and comparison for each repo  
  3. Final choice and rationale  
  4. How to extend the chosen repo  
  5. Effects of every parameter tweak  
  6. Beyond tuning: other optimization tricks  

Inline code comments help understanding.  
To run it, install **PyTorch3D** and **torch**:

```bash
conda create -n pytorch3d python=3.9
conda activate pytorch3d
pip install torch==2.4.0 torchvision==0.19 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```

Before training, rename all **.MTL** and **.TGA** files to lowercase extensions:  
```cmd
ren <path> *.MTL *.mtl
ren <path> *.TGA *.tga
```
