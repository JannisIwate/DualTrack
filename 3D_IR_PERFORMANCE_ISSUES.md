# 3D Image Registration Performance Issues - Analysis

## Critical Issues Found

### 1. **UNBOUNDED VOLUME ALLOCATION** ⚠️ CRITICAL
**Location:** `build_volume_from_slices()` (sitk.py, lines 457-490)

**Problem:**
The volume bounding box is computed from ALL corner points of ALL frames after transformation:
```python
volume_size = np.ceil((world_max - world_min) / spacing).astype(int) + 1
```

With 8 frames and typical motion (e.g., 5-10mm translation), the bounding box can easily expand far beyond what you'd expect:
- Frame size: 640×480 pixels → ~146×106 mm in world coords
- If frames move 50mm apart across the window, the bounding box becomes ~200×150×200mm
- At spacing of ~0.23mm, this is **~870 × 650 × ~same** = **~580 million voxels**
- With loose motion or off-plane rotations, this grows even larger

**Direct cause of crashes:** 
```python
volume_np = np.zeros((z, y, x), dtype=np.float32)  # Allocates 2.3 GB for 580M voxels!
weight_np = np.zeros_like(volume_np)               # Another 2.3 GB
```

Plus all the intermediate arrays in bincount operations.

---

### 2. **MASSIVE MEMORY ALLOCATION IN BINCOUNT** ⚠️ CRITICAL
**Location:** `build_volume_from_slices()` (sitk.py, lines 526-530)

**Problem:**
```python
total = volume_size[0] * volume_size[1] * volume_size[2]
sums   = np.bincount(flat_idx, weights=values, minlength=total)  # Allocates 'total' floats!
counts = np.bincount(flat_idx, minlength=total)                   # Allocates 'total' ints!
```

For a large volume (580M voxels):
- `sums`: 580M × 8 bytes (float32) = **4.6 GB**
- `counts`: 580M × 8 bytes (int64) = **4.6 GB**

**Why it's inefficient:** Most voxels are empty (sparse volume). You're allocating space for ALL voxels but only filling ~1-5%. A sparse data structure would be vastly more efficient.

---

### 3. **LACK OF VOLUME SIZE VALIDATION**
**Location:** `register_3d()` (sitk.py, line 187)

**Problem:** No check that the volume is reasonable before building it:
```python
volume, _, _ = build_volume_from_slices(volume_frames, volume_poses)
# ^ This can silently allocate 5-10GB+ before you know something's wrong
```

**Suggested fix:** Add early validation:
```python
# Before building volume
volume_size = compute_expected_volume_size(volume_frames, volume_poses, spacing)
if np.prod(volume_size) > MAX_VOXELS:
    raise ValueError(f"Volume too large: {np.prod(volume_size)} voxels exceeds {MAX_VOXELS}")
```

---

### 4. **EXPENSIVE HOLE FILLING ON LARGE VOLUMES**
**Location:** `_fill_interior_holes()` (sitk.py, lines 1004-1037)

**Problem:**
- `BinaryFillhole()` is O(n) for all voxels
- Multiple iterations with dilation filters add significant overhead
- On a 580M voxel volume, this is expensive even if sparse

**Also:** The hole filling step may not be necessary for registration—it's mainly for visualization/analysis. Registration can work with sparse volumes.

---

### 5. **AGGRESSIVE SITK OPTIMIZATION PARAMETERS**
**Location:** `build_registration_object()` (sitk.py, line 717)

```python
registration.SetOptimizerAsRegularStepGradientDescent(
    learningRate=0.1,
    minStep=1e-4,
    numberOfIterations=200  # ← 200 iterations on a huge volume!
)
```

**Problem:** With a very large volume, each metric evaluation is slow. 200 iterations × slow evaluation = 10+ seconds easily.

---

## Why Crashes Happen

When you process 8 frames:
1. Build volume → Allocates 2-10GB depending on motion
2. Run bincount → Allocates another 5-10GB
3. Garbage collection pressure → System starts swapping
4. SITK registration on huge volume → Each iteration is slow
5. Eventually OOM killer terminates the process

---

## Recommended Fixes

### Fix 1: Add Volume Size Validation (High Priority)
```python
def build_volume_from_slices(frames, poses, volume_spacing=(SPACING_X, SPACING_Y, SPACING_X)):
    # ... existing code ...
    
    volume_size = np.ceil((world_max - world_min) / spacing).astype(int) + 1
    
    # ADD THIS:
    max_voxels = 100_000_000  # ~400MB per array
    total_voxels = np.prod(volume_size)
    if total_voxels > max_voxels:
        print(f"WARNING: Volume size {volume_size} = {total_voxels} voxels")
        print(f"Downsampling volume by factor of {total_voxels / max_voxels:.1f}x")
        spacing = tuple(s * np.cbrt(total_voxels / max_voxels) for s in volume_spacing)
        volume_size = np.ceil((world_max - world_min) / spacing).astype(int) + 1
```

### Fix 2: Use Sparse Data Structures (Medium Priority)
Instead of dense arrays, use scipy sparse arrays or a dict-based approach:
```python
# Instead of:
volume_np = np.zeros((z, y, x), dtype=np.float32)

# Use:
from scipy.sparse import csr_matrix
volume_data = {}  # dict of {(x,y,z): value}
# Or use COO (coordinate) format for efficient construction
```

### Fix 3: Reduce Optimizer Iterations (Quick Win)
```python
# In build_registration_object():
registration.SetOptimizerAsRegularStepGradientDescent(
    learningRate=0.1,
    minStep=1e-4,
    numberOfIterations=50  # Reduce from 200
)
```

### Fix 4: Skip Hole Filling or Make It Optional (Medium Priority)
```python
# In build_volume_from_slices():
if "fill_holes" in options:  # Add optional flag
    volume = _fill_interior_holes(volume, mask_image)
else:
    # Just return sparse volume as-is
    pass
```

### Fix 5: Enable Multi-resolution Registration (Easy)
Already in the code but commented out! (sitk.py, line 724-729)
```python
if "multi_resolution" in options:
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
```

This dramatically reduces computation time by doing coarse registration first, then fine-tuning.

---

## Why Your System Crashes

**Scenario 1: OOM (Out of Memory)**
- Build volume: 8GB allocated
- Bincount sums/counts: 10GB more allocated
- Total: 18GB on a system with 16GB RAM
- → Swap thrashing → Crash

**Scenario 2: Single Bad Frame**
- If one frame has very different pose, the bounding box explodes
- Volume allocation explodes
- System crashes immediately

**Scenario 3: Slow Registration**
- Large volume with 200 iterations
- Each iteration: 580M voxel metric evaluation
- 200 × slow = 10+ seconds
- If registration gets stuck in local minimum, never converges

---

## Validation Checks to Add

Add these to [image_registration.py](image_registration.py#L96) before calling `register_3d()`:

```python
def validate_registration_input(window, pred_acc, config):
    """Check that registration will not crash/hang."""
    
    # Check window size
    if len(window) != len(pred_acc):
        raise ValueError(f"Window size {len(window)} != poses {len(pred_acc)}")
    
    # Estimate volume size
    volume_poses = pred_acc @ np.linalg.inv(pred_acc[0])
    
    # Get frame corners and compute bounds
    h, w = window[0].shape
    corners = np.array([[0, 0, 0], [w, 0, 0], [0, h, 0], [w, h, 0], [1, 1, 1]])
    
    world_pts = []
    for pose in volume_poses:
        pts = (pose @ np.pad(corners, ((0,0), (0,1)), constant_values=1).T).T[:, :3]
        world_pts.extend(pts)
    
    world_pts = np.array(world_pts)
    bounds = world_pts.max(axis=0) - world_pts.min(axis=0)
    
    # Estimate voxel count
    spacing = SPACING_X
    estimated_voxels = np.prod(bounds / spacing)
    
    if estimated_voxels > 200_000_000:
        print(f"⚠️  WARNING: Estimated {estimated_voxels:.0e} voxels (bounds: {bounds})")
        print(f"   This will likely crash or take 10+ seconds")
        return False
    
    return True
```

---

## Summary Table

| Issue | Severity | Memory Impact | Speed Impact | Fix Difficulty |
|-------|----------|---------------|--------------|-----------------|
| Unbounded volume | CRITICAL | 5-10GB | N/A (crashes first) | Easy |
| Bincount allocation | CRITICAL | 5-10GB | N/A (crashes first) | Medium |
| No validation | HIGH | Cascades above | Cascades above | Easy |
| Hole filling | MEDIUM | 1GB | +1-2 seconds | Easy |
| 200 iterations | MEDIUM | N/A | +5-10 seconds | Easy |
| Dense vs sparse | HIGH | 10x reduction possible | N/A | Hard |

---

## Immediate Action Items

1. **TODAY:** Add volume size validation (Fix 1) - prevents crashes
2. **TODAY:** Reduce optimizer iterations from 200 to 50 (Fix 3) - reduces time by ~4x
3. **TODAY:** Add optional hole filling flag (Fix 4) - skip unnecessary processing
4. **THIS WEEK:** Implement sparse volume option (Fix 2) - huge memory/speed win
5. **THIS WEEK:** Enable and tune multi-resolution mode (Fix 5) - better convergence + speed

Start with fixes 1, 3, 4—they're easy and will likely solve your immediate problems.
