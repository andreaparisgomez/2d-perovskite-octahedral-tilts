# 2D Perovskite Octahedral Tilt Calculator
Python script to compute octahedral tilt descriptors in layered metal-halide perovskites from crystallographic CIF files.
The script reports two geometric descriptors:

- **θ (M–X–M)** = metal–halide–metal bridging angle  
- **φ (X–X–X)** = halide–halide–halide rhomboid angle (~150°)
<img width="451" height="241" alt="image" src="https://github.com/user-attachments/assets/961a3977-e01c-4d3c-978c-69760c9b53df" />

These angles serve as geometric descriptors of octahedral tilting in 2D hybrid perovskite structures.

## Requirements

Python 3.9+

Required packages:
  numpy 
  pymatgen
  
Install with:
  pip install numpy pymatgen

## Usage

Run the script with a CIF file:
  python tiltcalc.py structure.cif


Example output:
θ (M–X–M) = 149.86°
φ (X–X–X) = 145.24 ± 0.03°


## Notes

The script automatically detects metal and halide atoms from the CIF file.  
Users can override the automatic detection using command-line options.

The method is designed primarily for layered (2D) organic-inorganic perovskite structures.
