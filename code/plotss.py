import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the logs - all four HIDDEN FTS runs with mask_fraction=0.5
data = {
    "32 (early stop at 20)": {
        "epochs": list(range(1, 21)),
        "loss": [
            52.8283, 20.1438, 13.8610, 10.8297, 9.5308, 8.9319, 8.5546, 8.3028,
            8.1452, 7.9337, 7.8092, 7.7195, 7.6426, 7.5799, 7.4438, 7.3948,
            7.3598, 7.3237, 7.1701, 7.1393
        ],
        "train_gdt": [
            0.6284, 0.7872, 0.8450, 0.8771, 0.8877, 0.8946, 0.8991, 0.9014,
            0.9014, 0.9034, 0.9046, 0.9066, 0.9075, 0.9083, 0.9088, 0.9095,
            0.9093, 0.9100, 0.9109, 0.9106
        ],
        "val_gdt": [
            0.7579, 0.8316, 0.8903, 0.8847, 0.9172, 0.9259, 0.9257, 0.9259,
            0.9308, 0.9394, 0.9373, 0.9362, 0.9393, 0.9345, 0.9373, 0.9350,
            0.9355, 0.9375, 0.9340, 0.9388
        ]
    },
    "64 (full 50 epochs)": {
        "epochs": list(range(1, 51)),
        "loss": [
            47.4421, 16.2732, 11.4487, 9.7931, 9.0048, 8.5477, 8.2888, 8.0123,
            7.8287, 7.6821, 7.5729, 7.4513, 7.3256, 7.2467, 7.1309, 7.0080,
            6.9633, 6.9222, 6.8660, 6.8042, 6.7412, 6.7199, 6.6319, 6.5621,
            6.4887, 6.4670, 6.4347, 6.3747, 6.3473, 6.3284, 6.3025, 6.2553,
            6.1552, 6.1510, 6.0901, 6.0845, 6.0542, 6.0411, 6.0019, 5.9763,
            5.9620, 5.9189, 5.9416, 5.8775, 5.8890, 5.8799, 5.8509, 5.9132,
            5.8629, 5.8534
        ],
        "train_gdt": [
            0.6534, 0.8200, 0.8693, 0.8858, 0.8947, 0.8986, 0.9011, 0.9032,
            0.9046, 0.9048, 0.9056, 0.9068, 0.9076, 0.9090, 0.9100, 0.9102,
            0.9116, 0.9119, 0.9125, 0.9126, 0.9137, 0.9138, 0.9143, 0.9143,
            0.9153, 0.9147, 0.9150, 0.9157, 0.9165, 0.9167, 0.9173, 0.9175,
            0.9172, 0.9185, 0.9191, 0.9199, 0.9200, 0.9199, 0.9195, 0.9203,
            0.9207, 0.9214, 0.9214, 0.9218, 0.9220, 0.9217, 0.9214, 0.9219,
            0.9218, 0.9221
        ],
        "val_gdt": [
            0.8016, 0.8751, 0.8923, 0.9158, 0.9172, 0.9262, 0.9279, 0.9337,
            0.9300, 0.9322, 0.9327, 0.9350, 0.9345, 0.9393, 0.9361, 0.9357,
            0.9384, 0.9388, 0.9386, 0.9382, 0.9374, 0.9387, 0.9329, 0.9427,
            0.9390, 0.9388, 0.9378, 0.9421, 0.9437, 0.9410, 0.9410, 0.9447,
            0.9399, 0.9425, 0.9419, 0.9444, 0.9441, 0.9450, 0.9436, 0.9407,
            0.9475, 0.9478, 0.9431, 0.9470, 0.9461, 0.9445, 0.9462, 0.9463,
            0.9468, 0.9468
        ]
    },
    "128 (best at epoch 33, early stop)": {
        "epochs": list(range(1, 44)),
        "loss": [
            57.4177, 23.5110, 17.0406, 13.0586, 11.0085, 9.7881, 9.3247, 8.9044,
            8.6397, 8.4185, 8.2093, 8.0896, 7.9934, 7.8023, 7.7046, 7.5895,
            7.5696, 7.5098, 7.3533, 7.2871, 7.2910, 7.2472, 7.1463, 7.0871,
            6.9970, 7.0251, 6.9934, 7.0203, 7.0662, 6.8206, 6.7452, 6.7426,
            6.6396, 6.6449, 6.6397, 6.6202, 6.6121, 6.6867, 6.5837, 6.5220,
            6.5454, 6.6770, 6.5551
        ],
        "train_gdt": [
            0.6108, 0.7576, 0.8147, 0.8522, 0.8738, 0.8857, 0.8914, 0.8959,
            0.8971, 0.8980, 0.8998, 0.8999, 0.9019, 0.9023, 0.9044, 0.9045,
            0.9048, 0.9053, 0.9073, 0.9065, 0.9071, 0.9086, 0.9092, 0.9088,
            0.9106, 0.9100, 0.9105, 0.9109, 0.9105, 0.9115, 0.9117, 0.9117,
            0.9127, 0.9134, 0.9125, 0.9130, 0.9120, 0.9136, 0.9126, 0.9131,
            0.9139, 0.9131, 0.9138
        ],
        "val_gdt": [
            0.7330, 0.8189, 0.8633, 0.8874, 0.9038, 0.9117, 0.9124, 0.9223,
            0.9217, 0.9236, 0.9267, 0.9291, 0.9255, 0.9278, 0.9246, 0.9377,
            0.9400, 0.9396, 0.9336, 0.9327, 0.9358, 0.9374, 0.9226, 0.9341,
            0.9326, 0.9422, 0.9262, 0.9366, 0.9329, 0.9363, 0.9393, 0.9391,
            0.9448, 0.9325, 0.9361, 0.9421, 0.9386, 0.9345, 0.9391, 0.9438,
            0.9345, 0.9347, 0.9312
        ]
    },
    "256 (best at epoch 42)": {
        "epochs": list(range(1, 51)),
        "loss": [
            35.8185, 11.8809, 10.0976, 9.0634, 8.7904, 8.4795, 8.2510, 8.0328,
            8.0705, 7.7548, 7.6831, 7.6163, 7.4703, 7.3751, 7.2917, 7.3266,
            7.2247, 7.1551, 7.1670, 7.0261, 7.0395, 6.9421, 6.9383, 6.8769,
            6.7863, 6.7585, 6.6660, 6.6363, 6.5802, 6.5524, 6.4696, 6.4324,
            6.4321, 6.3396, 6.2936, 6.2252, 6.2173, 6.1931, 6.1423, 6.0838,
            6.0547, 6.0463, 6.0091, 6.0018, 5.9622, 5.9995, 5.9338, 5.9622,
            5.9300, 5.9163
        ],
        "train_gdt": [
            0.7135, 0.8620, 0.8852, 0.8932, 0.8943, 0.8971, 0.8979, 0.8992,
            0.9008, 0.9036, 0.9038, 0.9033, 0.9054, 0.9064, 0.9073, 0.9064,
            0.9084, 0.9086, 0.9099, 0.9098, 0.9095, 0.9112, 0.9111, 0.9121,
            0.9130, 0.9130, 0.9135, 0.9145, 0.9145, 0.9146, 0.9152, 0.9162,
            0.9165, 0.9175, 0.9183, 0.9184, 0.9189, 0.9184, 0.9197, 0.9211,
            0.9206, 0.9208, 0.9211, 0.9217, 0.9217, 0.9211, 0.9222, 0.9220,
            0.9221, 0.9217
        ],
        "val_gdt": [
            0.8558, 0.9019, 0.9136, 0.9223, 0.9159, 0.9093, 0.9324, 0.9337,
            0.9210, 0.9268, 0.9284, 0.9297, 0.9306, 0.9366, 0.9376, 0.9329,
            0.9388, 0.9400, 0.9385, 0.9353, 0.9385, 0.9332, 0.9425, 0.9420,
            0.9399, 0.9431, 0.9407, 0.9437, 0.9401, 0.9426, 0.9368, 0.9455,
            0.9408, 0.9419, 0.9429, 0.9459, 0.9450, 0.9444, 0.9457, 0.9439,
            0.9441, 0.9470, 0.9450, 0.9447, 0.9445, 0.9441, 0.9451, 0.9457,
            0.9457, 0.9457
        ]
    }
}

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Colors for each run
colors = ['blue', 'green', 'red', 'purple']
markers = ['o', 's', '^', 'D']
run_labels = ['h = 32', 'h = 64', 
              'h = 128', 'h = 256']

for ax, (ylabel, key) in zip(axes, [("Training Loss", "loss"),
                                     ("Training GDT-TS", "train_gdt"),
                                     ("Validation GDT-TS", "val_gdt")]):
    for (run_name, vals), color, marker, label in zip(data.items(), colors, markers, run_labels):
        ax.plot(vals["epochs"], vals[key],
                label=label,
                color=color,
                marker=marker,
                markevery=5,  # show markers every 5 epochs
                linestyle='-',
                linewidth=2,
                markersize=6)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f"{ylabel} vs. Epoch", fontsize=14)

plt.tight_layout()
plt.savefig("hidden_fts_all_runs_plots.png", dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*90)
print("SUMMARY STATISTICS - ALL HIDDEN FTS RUNS (mask_fraction=0.5)")
print("="*90)
print(f"{'Run':<25} {'Best Epoch':<12} {'Val GDT':<12} {'Train GDT':<12} {'Loss':<12} {'Checkpoint'}")
print("-"*90)

for run_name, vals in data.items():
    best_idx = np.argmax(vals["val_gdt"])
    best_epoch = vals["epochs"][best_idx]
    best_val = vals["val_gdt"][best_idx]
    best_train = vals["train_gdt"][best_idx]
    best_loss = vals["loss"][best_idx]
    
    # Truncate run name for display
    display_name = run_name[:24]
    print(f"{display_name:<25} {best_epoch:<12} {best_val:<12.4f} {best_train:<12.4f} {best_loss:<12.4f} ./checkpoints/r8_l16_m0.5.pt")

print("-"*90)

# Find overall best
all_best = []
for run_name, vals in data.items():
    best_idx = np.argmax(vals["val_gdt"])
    all_best.append((run_name, vals["epochs"][best_idx], vals["val_gdt"][best_idx]))

overall_best = max(all_best, key=lambda x: x[2])
print(f"\nOverall Best: {overall_best[0]} at epoch {overall_best[1]} with Val GDT = {overall_best[2]:.4f}")

# Additional analysis
print("\n" + "="*90)
print("KEY OBSERVATIONS")
print("="*90)
print("• All runs use radius=8.0A, n_layers=16, mask_fraction=0.5")
print("• Different random seeds/initializations lead to different convergence patterns")
print("• Best validation GDT achieved: 0.9478 (Run 2, epoch 42)")
print("• Run 1 (early stopped at epoch 20) reached 0.9394 (epoch 10)")
print("• Run 3 (early stopped at epoch 33) reached 0.9448 (epoch 33)")
print("• Run 4 achieved 0.9470 (epoch 42)")
print("\n• The model shows consistent performance around 0.94-0.948 Val GDT")
print("• Early stopping at epoch 42 appears to give the best balance")
print("• All runs converge to similar performance levels despite different initialization")