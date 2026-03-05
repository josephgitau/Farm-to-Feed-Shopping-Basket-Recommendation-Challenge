"""
Triple Blend: V2 + (V4+V11)

Strategy:
- V4_V11 base: V11 for WAUC, V4 for MAE
- Blend V2 into WAUC predictions with weights

Weights:
- T1: 20% V2, 80% V4_V11 (V11's purchase)
- T2: 20% V2, 80% V4_V11 (V11's purchase)
- Quantity: 100% V4

Output: submission_blend_v2_v4_v11.csv
"""

import pandas as pd
import numpy as np

# Weights
T1_V2_WEIGHT = 0.35      # V2 weight for T1 WAUC
T2_V4_11_WEIGHT = 0.75   # V4_V11 (V11) weight for T2 WAUC

# Calculate complementary weights
T1_V4_11_WEIGHT = 1 - T1_V2_WEIGHT  # 0.8
T2_V2_WEIGHT = 1 - T2_V4_11_WEIGHT  # 0.2

# Load submissions
v2 = pd.read_csv('submission_v2.csv')
v4 = pd.read_csv('submission_v4.csv')
v11 = pd.read_csv('submission_v11.csv')

print(f"V2 shape: {v2.shape}")
print(f"V4 shape: {v4.shape}")
print(f"V11 shape: {v11.shape}")

# Create blend
blend = v2[['ID']].copy()

# WAUC T1: Blend V2 (20%) + V11 (80%)
blend['Target_purchase_next_1w'] = (
    T1_V2_WEIGHT * v2['Target_purchase_next_1w'] + 
    T1_V4_11_WEIGHT * v11['Target_purchase_next_1w']
)

# WAUC T2: Blend V2 (20%) + V11 (80%)
blend['Target_purchase_next_2w'] = (
    T2_V2_WEIGHT * v2['Target_purchase_next_2w'] + 
    T2_V4_11_WEIGHT * v11['Target_purchase_next_2w']
)

# Quantity: Use V4 (100% for MAE)
blend['Target_qty_next_1w'] = v4['Target_qty_next_1w']
blend['Target_qty_next_2w'] = v4['Target_qty_next_2w']

# Save
blend.to_csv('submission_blend_v2_v4_v11.csv', index=False)
print("\nSaved: submission_blend_v2_v4_v11.csv")

# Summary
print("\n" + "="*60)
print("BLEND SUMMARY")
print("="*60)
print(f"Target_purchase_next_1w: {T1_V2_WEIGHT:.0%} V2 + {T1_V4_11_WEIGHT:.0%} V11")
print(f"Target_purchase_next_2w: {T2_V2_WEIGHT:.0%} V2 + {T2_V4_11_WEIGHT:.0%} V11")
print(f"Target_qty_next_1w:      100% V4")
print(f"Target_qty_next_2w:      100% V4")
print("="*60)

# Stats
print("\nBlend Prediction Statistics:")
print(f"Target_purchase_next_1w: mean={blend['Target_purchase_next_1w'].mean():.4f}, std={blend['Target_purchase_next_1w'].std():.4f}")
print(f"Target_purchase_next_2w: mean={blend['Target_purchase_next_2w'].mean():.4f}, std={blend['Target_purchase_next_2w'].std():.4f}")
print(f"Target_qty_next_1w:      mean={blend['Target_qty_next_1w'].mean():.4f}, std={blend['Target_qty_next_1w'].std():.4f}")
print(f"Target_qty_next_2w:      mean={blend['Target_qty_next_2w'].mean():.4f}, std={blend['Target_qty_next_2w'].std():.4f}")
