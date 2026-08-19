import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))  # keep same figsize as before

ax.bar(bins, counts, color='#1f77b4')  # your existing bar call

ax.set_xlabel('Missing residue percentage', fontsize=14)
ax.set_ylabel('Number of complexes', fontsize=14)

ax.tick_params(axis='both', labelsize=12)  # bigger tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # keep your rotation

plt.tight_layout()
plt.savefig('missing_residue_hist.png', dpi=150)