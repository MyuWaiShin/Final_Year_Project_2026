# Data Collection Guide for Object Detection

## Why Retrain?

Your current models are misclassifying objects because:
1. **Limited training data** - Not enough variety in angles, lighting, distances
2. **White background only** - Model overfitted to white table, struggles with blue ground
3. **Fixed camera position** - No variation in viewpoint/height

## YOLOv8n vs YOLO26n

### YOLOv8n (Recommended ✅)
- **Standard Ultralytics YOLO** architecture
- Better community support and documentation
- More stable for deployment
- Compatible with most conversion tools
- **Use this for your project**

### YOLO26n (Not Recommended ❌)
- Custom/experimental YOLO variant
- Less stable, fewer resources
- Harder to convert to deployment formats
- Not worth the complexity for this use case

**Recommendation:** Stick with **YOLOv8n** for training. It's proven, stable, and works well for your 3-class problem.

---

## Data Collection Strategy

### Setup
- **Camera**: OAK-D Lite mounted on UR10 end-effector (hand-eye configuration)
- **Objects**: Cubes, cylinders, arcs (styrofoam, vivid colors)
- **Background**: Blue ground (matches deployment environment)
- **Lighting**: Varied (natural + artificial)

### Key Principles

#### 1. **Viewpoint Diversity** (Most Important!)
Your robot will see objects from different heights and angles during operation.

**Collect data at multiple heights:**
- **High (80-100cm)**: Bird's eye view, objects appear smaller
- **Medium (50-70cm)**: Standard working height
- **Low (30-40cm)**: Close-up, objects appear larger

**Collect data at multiple angles:**
- Top-down (0°)
- Angled (30°, 45°, 60°)
- Side views (if robot approaches from different directions)

#### 2. **Object Arrangements**
- **Single objects**: Each object alone
- **Pairs**: Two objects close together
- **Groups**: 3-5 objects in various configurations
- **Occlusions**: Objects partially blocking each other
- **Distances**: Objects at different distances from camera

#### 3. **Lighting Variations**
- Bright overhead lighting
- Dim lighting
- Shadows from robot arm
- Different times of day (if near windows)

#### 4. **Object Variations**
- Different orientations (rotated, tilted)
- Different positions in frame (center, edges, corners)
- Different colors of same shape (if you have multiple)

---

## Recommended Collection Method

### Option 1: Manual Collection (More Control)
**Best for initial dataset**

1. **Setup positions manually**:
   - Place objects in different arrangements on blue ground
   - Move UR10 to different heights (high, medium, low)
   - Rotate end-effector to different angles

2. **Record 10-20 second videos** at each position
3. **Extract frames** from videos (your existing pipeline)

**Pros:**
- Full control over data quality
- Ensures good coverage
- Can verify each setup

**Cons:**
- Time-consuming
- Requires manual positioning

---

### Option 2: Automated Random Collection (More Data, Less Control)
**Good for augmenting dataset**

**Strategy:**
1. **Place objects randomly** on blue ground (3-5 objects)
2. **Program UR10 to move randomly**:
   - Random heights (30-100cm)
   - Random XY positions (within safe workspace)
   - Random wrist rotations (±30°)
3. **Record continuously** while robot moves
4. **Extract frames** at regular intervals (every 0.5-1 second)

**Pros:**
- Collects lots of varied data quickly
- Natural motion blur (good for robustness)
- Less manual work

**Cons:**
- Some frames may be blurry
- Less control over specific scenarios
- Need to filter bad frames

---

## Hybrid Approach (Recommended ✅)

**Combine both methods:**

### Phase 1: Manual Collection (70% of data)
- 5-10 setups per object type
- 3 heights × 3 angles = 9 positions per setup
- ~10 second video per position
- **Goal**: ~500-800 high-quality frames

### Phase 2: Automated Collection (30% of data)
- 3-5 random object arrangements
- UR10 random movements for 2-3 minutes each
- Extract frames every 0.5 seconds
- **Goal**: ~200-400 varied frames

### Phase 3: Validation Set
- **Separate session** with different lighting/setup
- Manual collection only
- **Goal**: ~100-150 frames for validation

---

## Data Collection Checklist

### Before Recording
- [ ] Clean blue ground (no debris)
- [ ] Check camera focus and exposure
- [ ] Verify UR10 workspace limits (safety)
- [ ] Test recording script (see below)

### During Recording
- [ ] Record at 3 different heights minimum
- [ ] Include objects at edges of frame
- [ ] Include partial occlusions
- [ ] Vary lighting if possible
- [ ] Record some "empty" frames (no objects) for background

### After Recording
- [ ] Extract frames from videos
- [ ] Remove blurry/bad frames
- [ ] Split: 80% train, 10% val, 10% test
- [ ] Label using your existing tool
- [ ] Verify labels are correct

---

## Expected Dataset Size

For good performance with 3 classes:
- **Minimum**: 500 labeled images (166 per class)
- **Recommended**: 1000-1500 labeled images (333-500 per class)
- **Ideal**: 2000+ labeled images (666+ per class)

More data = better generalization!

---

## UR10 Movement Pattern Suggestion

For automated collection, program the UR10 to:

```python
# Pseudocode for UR10 movement
heights = [0.3, 0.5, 0.7, 0.9]  # meters above ground
angles = [-30, 0, 30]  # degrees tilt

for each object_arrangement:
    for height in heights:
        for angle in angles:
            move_to_position(x=random, y=random, z=height, tilt=angle)
            wait(0.5 seconds)  # stabilize
            # Camera records continuously
```

This ensures systematic coverage while still having randomness in XY position.

---

## Next Steps

1. **Create recording script** (see `record_data.py` below)
2. **Collect Phase 1 data** (manual, high quality)
3. **Label and train initial model**
4. **Evaluate performance**
5. **Collect Phase 2 data** (automated, if needed)
6. **Retrain and deploy**
