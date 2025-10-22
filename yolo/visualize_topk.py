"""
topk.txt의 유사도 분포 시각화
"""
import re
import matplotlib.pyplot as plt
import numpy as np
import os

# topk.txt 파일 읽기
debug_dir = os.path.join(os.path.dirname(__file__), 'debug_sim')
topk_file = os.path.join(debug_dir, 'topk.txt')

with open(topk_file, 'r') as f:
    lines = f.readlines()

# 유사도 값 추출
abs_scores = []
rel_scores = []

for line in lines:
    line = line.strip()
    if not line:
        continue

    # ABS와 REL 분리
    if '| ABS top' in line:
        # j=80:0.48, j=38:0.43 형식에서 숫자 추출
        matches = re.findall(r':(\d+\.\d+)', line)
        abs_scores.extend([float(m) for m in matches])
    elif '| REL top' in line:
        matches = re.findall(r':(\d+\.\d+)', line)
        rel_scores.extend([float(m) for m in matches])

# min_similarity 값 (기본값 0.7)
min_similarity = 0.7

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ABS 히스토그램
ax1 = axes[0, 0]
ax1.hist(abs_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(min_similarity, color='red', linestyle='--', linewidth=2, label=f'min_similarity={min_similarity}')
ax1.set_xlabel('Similarity Score (ABS)')
ax1.set_ylabel('Frequency')
ax1.set_title('Absolute Similarity Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 통계 정보 추가
below_threshold_abs = sum(1 for s in abs_scores if s < min_similarity)
total_abs = len(abs_scores)
ax1.text(0.02, 0.95, f'Below threshold: {below_threshold_abs}/{total_abs} ({below_threshold_abs/total_abs*100:.1f}%)',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. REL 히스토그램
ax2 = axes[0, 1]
ax2.hist(rel_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
ax2.axvline(min_similarity, color='red', linestyle='--', linewidth=2, label=f'min_similarity={min_similarity}')
ax2.set_xlabel('Similarity Score (REL)')
ax2.set_ylabel('Frequency')
ax2.set_title('Relative Similarity Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 통계 정보 추가
below_threshold_rel = sum(1 for s in rel_scores if s < min_similarity)
total_rel = len(rel_scores)
ax2.text(0.02, 0.95, f'Below threshold: {below_threshold_rel}/{total_rel} ({below_threshold_rel/total_rel*100:.1f}%)',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. ABS 누적 분포 (CDF)
ax3 = axes[1, 0]
sorted_abs = np.sort(abs_scores)
cdf_abs = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
ax3.plot(sorted_abs, cdf_abs, linewidth=2, color='blue')
ax3.axvline(min_similarity, color='red', linestyle='--', linewidth=2, label=f'min_similarity={min_similarity}')
ax3.set_xlabel('Similarity Score (ABS)')
ax3.set_ylabel('Cumulative Probability')
ax3.set_title('Absolute Similarity - Cumulative Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# min_similarity에서의 누적 확률
cdf_at_threshold_abs = np.searchsorted(sorted_abs, min_similarity) / len(sorted_abs)
ax3.text(0.02, 0.95, f'CDF at threshold: {cdf_at_threshold_abs*100:.1f}%',
         transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. REL 누적 분포 (CDF)
ax4 = axes[1, 1]
sorted_rel = np.sort(rel_scores)
cdf_rel = np.arange(1, len(sorted_rel) + 1) / len(sorted_rel)
ax4.plot(sorted_rel, cdf_rel, linewidth=2, color='green')
ax4.axvline(min_similarity, color='red', linestyle='--', linewidth=2, label=f'min_similarity={min_similarity}')
ax4.set_xlabel('Similarity Score (REL)')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('Relative Similarity - Cumulative Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

# min_similarity에서의 누적 확률
cdf_at_threshold_rel = np.searchsorted(sorted_rel, min_similarity) / len(sorted_rel)
ax4.text(0.02, 0.95, f'CDF at threshold: {cdf_at_threshold_rel*100:.1f}%',
         transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 저장
output_path = os.path.join(debug_dir, 'similarity_distribution.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Similarity distribution plot saved to: {output_path}")

# 통계 출력
print("\n=== Absolute Similarity Statistics ===")
print(f"Total scores: {total_abs}")
print(f"Below threshold ({min_similarity}): {below_threshold_abs} ({below_threshold_abs/total_abs*100:.1f}%)")
print(f"Above threshold: {total_abs - below_threshold_abs} ({(total_abs - below_threshold_abs)/total_abs*100:.1f}%)")
print(f"Min: {min(abs_scores):.3f}, Max: {max(abs_scores):.3f}")
print(f"Mean: {np.mean(abs_scores):.3f}, Median: {np.median(abs_scores):.3f}")
print(f"Std: {np.std(abs_scores):.3f}")

print("\n=== Relative Similarity Statistics ===")
print(f"Total scores: {total_rel}")
print(f"Below threshold ({min_similarity}): {below_threshold_rel} ({below_threshold_rel/total_rel*100:.1f}%)")
print(f"Above threshold: {total_rel - below_threshold_rel} ({(total_rel - below_threshold_rel)/total_rel*100:.1f}%)")
print(f"Min: {min(rel_scores):.3f}, Max: {max(rel_scores):.3f}")
print(f"Mean: {np.mean(rel_scores):.3f}, Median: {np.median(rel_scores):.3f}")
print(f"Std: {np.std(rel_scores):.3f}")

plt.show()
