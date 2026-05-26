import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import skew


# ============================================================
# ACTIVATION + GRADIENT + CHANNEL ANALYSIS FOR CNN1D
# ============================================================

class ActivationAnalyzer:
    """
    Comprehensive CNN activation diagnostics.

    Measures:
    - Activation growth
    - Vanishing/exploding activations
    - Dead neurons
    - Per-channel activity
    - Activation distributions
    - Gradient flow
    - Dominant channels
    """

    def __init__(self, model, device=None):

        self.model = model

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.device = device
        self.model.to(device)

        self.activation_storage = defaultdict(list)
        self.gradient_storage = defaultdict(list)

        self.forward_hooks = []
        self.backward_hooks = []

    # ============================================================
    # HOOKS
    # ============================================================

    def _make_forward_hook(self, name):

        def hook(module, inputs, output):

            if isinstance(output, tuple):
                output = output[0]

            output = output.detach().cpu()

            self.activation_storage[name].append(output)

        return hook

    def _make_backward_hook(self, name):

        def hook(module, grad_input, grad_output):

            grad = grad_output[0]

            if grad is not None:
                self.gradient_storage[name].append(
                    grad.detach().cpu()
                )

        return hook

    def register_hooks(self):

        valid_types = (
            nn.Conv1d,
            nn.Linear,
            nn.AdaptiveAvgPool1d,
            nn.MaxPool1d
        )

        for name, module in self.model.named_modules():

            if name == "":
                continue

            if isinstance(module, valid_types):
                continue

            # forward hook
            fh = module.register_forward_hook(
                self._make_forward_hook(name)
            )

            # backward hook
            bh = module.register_full_backward_hook(
                self._make_backward_hook(name)
            )

            self.forward_hooks.append(fh)
            self.backward_hooks.append(bh)

    def remove_hooks(self):

        for h in self.forward_hooks:
            h.remove()

        for h in self.backward_hooks:
            h.remove()

    # ============================================================
    # MAIN ANALYSIS
    # ============================================================

    def analyze(
        self,
        data_loader,
        criterion=None,
        n_batches=10
    ):

        self.activation_storage.clear()
        self.gradient_storage.clear()

        self.model.eval()

        self.register_hooks()

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        for i, (inputs, targets) in enumerate(data_loader):

            if i >= n_batches:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()

            outputs = self.model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

        self.remove_hooks()

        stats = self.compute_statistics()

        return stats

    # ============================================================
    # STATISTICS
    # ============================================================

    def compute_statistics(self):

        stats = {}

        for name in self.activation_storage.keys():

            activations = torch.cat(
                [x.flatten() for x in self.activation_storage[name]]
            ).numpy()

            abs_vals = np.abs(activations)

            # ----------------------------------------------------
            # channel-wise analysis
            # ----------------------------------------------------

            channel_mean = None
            channel_std = None
            dominant_channel = None

            sample_tensor = self.activation_storage[name][0]

            batch_channel_means = []

            for tensor in self.activation_storage[name]:

                # ------------------------------------------------
                # CNN feature maps: [B, C, T]
                # ------------------------------------------------

                if tensor.ndim == 3:

                    cm = (
                        tensor.abs()
                        .mean(dim=(0, 2))
                        .numpy()
                    )

                    batch_channel_means.append(cm)

                # ------------------------------------------------
                # Linear/features: [B, F]
                # ------------------------------------------------

                elif tensor.ndim == 2:

                    cm = (
                        tensor.abs()
                        .mean(dim=0)
                        .numpy()
                    )

                    batch_channel_means.append(cm)

                # ------------------------------------------------
                # Scalars / unusual tensors
                # ------------------------------------------------

                else:

                    continue

            # finalize stats

            if len(batch_channel_means) > 0:
                channel_means = np.mean(batch_channel_means, axis=0)

                channel_mean = channel_means

                channel_std = np.std(channel_means)

                dominant_channel = int(np.argmax(channel_means))

            # ----------------------------------------------------
            # gradient stats
            # ----------------------------------------------------

            if name in self.gradient_storage:

                grads = torch.cat(
                    [g.flatten() for g in self.gradient_storage[name]]
                ).numpy()

                grad_norm = np.sqrt(np.mean(grads ** 2))
                grad_mean = np.mean(np.abs(grads))

            else:

                grad_norm = 0
                grad_mean = 0

            # ----------------------------------------------------
            # final stats
            # ----------------------------------------------------

            stats[name] = {

                # activation stats
                "mean": float(np.mean(activations)),
                "std": float(np.std(activations)),
                "abs_mean": float(np.mean(abs_vals)),
                "max": float(np.max(abs_vals)),
                "min": float(np.min(abs_vals)),
                "l2_norm": float(
                    np.sqrt(np.mean(activations ** 2))
                ),

                # dead neurons
                "dead_ratio": float(
                    np.mean(abs_vals < 1e-6)
                ),

                "near_zero_ratio": float(
                    np.mean(abs_vals < 1e-3)
                ),

                # distribution shape
                "skewness": float(skew(activations)),

                # gradients
                "grad_norm": float(grad_norm),
                "grad_mean": float(grad_mean),

                # channel analysis
                "channel_mean": channel_mean,
                "channel_std": channel_std,
                "dominant_channel": dominant_channel,
            }

        return stats


# ============================================================
# PLOTTING
# ============================================================

def plot_full_analysis(stats, title="CNN Analysis", figsize=(18, 16)):

    layers = list(stats.keys())

    x = np.arange(len(layers))

    abs_mean = [stats[l]["abs_mean"] for l in layers]
    l2 = [stats[l]["l2_norm"] for l in layers]
    dead = [stats[l]["dead_ratio"] for l in layers]
    grad = [stats[l]["grad_norm"] for l in layers]
    skewness = [stats[l]["skewness"] for l in layers]

    fig, axes = plt.subplots(5, 1, figsize=figsize)

    # ========================================================
    # 1. Activation growth
    # ========================================================

    axes[0].plot(x, abs_mean, "o-", linewidth=2)

    axes[0].set_yscale("log")

    axes[0].set_title("Activation Growth")

    axes[0].set_ylabel("Mean |activation|")

    axes[0].axhline(
        y=1e-4,
        color="orange",
        linestyle="--",
        label="Vanishing"
    )

    axes[0].axhline(
        y=1e4,
        color="red",
        linestyle="--",
        label="Exploding"
    )

    axes[0].legend()

    # ========================================================
    # 2. L2 norm
    # ========================================================

    axes[1].plot(x, l2, "o-", linewidth=2)

    axes[1].set_yscale("log")

    axes[1].set_title("L2 Norm per Layer")

    axes[1].set_ylabel("L2 norm")

    # ========================================================
    # 3. Dead neurons
    # ========================================================

    axes[2].bar(x, dead)

    axes[2].set_ylim(0, 1)

    axes[2].set_title("Dead Neuron Ratio")

    axes[2].set_ylabel("Dead ratio")

    axes[2].axhline(
        y=0.5,
        color="red",
        linestyle="--"
    )

    # ========================================================
    # 4. Gradient flow
    # ========================================================

    axes[3].plot(x, grad, "o-", linewidth=2)

    axes[3].set_yscale("log")

    axes[3].set_title("Gradient Flow")

    axes[3].set_ylabel("Gradient norm")

    # ========================================================
    # 5. Distribution skewness
    # ========================================================

    axes[4].bar(x, skewness)

    axes[4].set_title("Activation Distribution Skewness")

    axes[4].set_ylabel("Skewness")

    # ========================================================

    for ax in axes:

        ax.set_xticks(x)

        ax.set_xticklabels(
            layers,
            rotation=45,
            ha="right",
            fontsize=8
        )

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()

    return fig


# ============================================================
# CHANNEL VISUALIZATION
# ============================================================

def plot_channel_activity(stats, layer_name):

    if stats[layer_name]["channel_mean"] is None:

        print(f"{layer_name} has no channel structure.")
        return

    channel_means = stats[layer_name]["channel_mean"]

    plt.figure(figsize=(10, 4))

    plt.bar(
        np.arange(len(channel_means)),
        channel_means
    )

    plt.title(f"Channel Activity — {layer_name}")

    plt.xlabel("Channel index")

    plt.ylabel("Mean |activation|")

    plt.grid(True, alpha=0.3)

    plt.show()


# ============================================================
# REPORT
# ============================================================

def print_detailed_report(stats):

    print("\n" + "=" * 140)

    print(
        f"{'Layer':<20}"
        f"{'AbsMean':>12}"
        f"{'L2':>12}"
        f"{'Grad':>12}"
        f"{'Dead%':>12}"
        f"{'Skew':>12}"
        f"{'DomCh':>12}"
        f"{'Status':>20}"
    )

    print("=" * 140)

    for layer, s in stats.items():

        issues = []

        # vanishing
        if s["abs_mean"] < 1e-4:
            issues.append("VANISHING")

        # exploding
        if s["abs_mean"] > 1e4:
            issues.append("EXPLODING")

        # dead neurons
        if s["dead_ratio"] > 0.5:
            issues.append("DEAD")

        # weak gradients
        if s["grad_norm"] < 1e-7:
            issues.append("WEAK_GRAD")

        if len(issues) == 0:
            issues = ["OK"]

        print(
            f"{layer:<20}"
            f"{s['abs_mean']:>12.4f}"
            f"{s['l2_norm']:>12.4f}"
            f"{s['grad_norm']:>12.4e}"
            f"{100*s['dead_ratio']:>11.1f}%"
            f"{s['skewness']:>12.2f}"
            f"{str(s['dominant_channel']):>12}"
            f"{','.join(issues):>20}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from Classification.cnn1D_model import CNN1D_Wide
    from utils.dataloader import stratified_group_split

    # --------------------------------------------------------
    # data
    # --------------------------------------------------------

    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"

    train_loader, val_loader, test_loader, dataset = (
        stratified_group_split(data_dir)
    )

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = CNN1D_Wide()

    model.load_state_dict(
        torch.load(
            "../cnn1d_model_new.ckpt",
            map_location="cpu"
        )
    )

    # --------------------------------------------------------
    # analysis
    # --------------------------------------------------------

    analyzer = ActivationAnalyzer(model)

    stats = analyzer.analyze(
        val_loader,
        n_batches=10
    )

    # --------------------------------------------------------
    # report
    # --------------------------------------------------------

    print_detailed_report(stats)

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------

    plot_full_analysis(
        stats,
        title="CNN1D_Wide Full Diagnostics"
    )

    # --------------------------------------------------------
    # inspect channels
    # --------------------------------------------------------

    # Example:
    plot_channel_activity(stats, "conv1")

    plot_channel_activity(stats, "conv2")

    plot_channel_activity(stats, "conv3")

    plot_channel_activity(stats, "conv4")