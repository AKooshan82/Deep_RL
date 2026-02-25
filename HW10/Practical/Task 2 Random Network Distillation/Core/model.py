# model.py
# This module defines the neural networks used in the RND+PPO agent.
# Students are expected to implement the TargetModel and PredictorModel architectures and their initialization.

from abc import ABC
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


# === Policy Network ===
class PolicyModel(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        c, w, h = state_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        flatten_size = 32 * 7 * 7

        self.fc1 = nn.Linear(flatten_size, 256)
        self.gru = nn.GRUCell(256, 256)

        self.extra_value_fc = nn.Linear(256, 256)
        self.extra_policy_fc = nn.Linear(256, 256)

        self.policy = nn.Linear(256, self.n_actions)
        self.int_value = nn.Linear(256, 1)
        self.ext_value = nn.Linear(256, 1)

        # Orthogonal initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs, hidden_state):
        if inputs.ndim == 5:
            inputs = inputs.squeeze(1)

        x = inputs / 255.
        x = self.conv(x)
        x = F.relu(self.fc1(x))
        h = self.gru(x, hidden_state)

        x_v = h + F.relu(self.extra_value_fc(h))
        x_pi = h + F.relu(self.extra_policy_fc(h))

        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)

        policy_logits = self.policy(x_pi)
        probs = F.softmax(policy_logits, dim=1)
        dist = Categorical(probs)

        return dist, int_value, ext_value, probs, h


# === Target Model ===
class TargetModel(nn.Module, ABC):
    """Lightweight convolutional encoder producing a 512‑D feature vector."""

    def __init__(self, state_shape: Tuple[int, int, int]):
        super().__init__()

        # Unpack the (channels, width, height) of the observation space
        c, w, h = state_shape

        # ------------------------------------------------------------------
        # Convolutional backbone
        # ------------------------------------------------------------------
        # Three stride‑2 conv layers reduce spatial resolution by 8× overall
        # while increasing the number of feature maps.
        self.conv1 = nn.Conv2d(c,   32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,  64, kernel_size=3, stride=2, padding=1)

        # After the conv stack the feature map is small (usually 1×1 or 2×2).
        # For an 84×84 input this flattening results in a 64‑element vector.
        conv_out_size = 64  # <-- adjust if you change the input resolution

        # Final linear layer projects to a 512‑D latent representation
        self.encoded_features = nn.Linear(conv_out_size, 512)

        # Apply orthogonal weight initialisation
        self._init_weights()

    # ----------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Orthogonal init for conv & linear layers (He gain)."""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    layer.bias.data.zero_()

    # ----------------------------------------------------------------------
    # Forward computation
    # ----------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into 512‑D feature vectors."""
        # Normalise pixel values from [0, 255] -> [0, 1]
        x = inputs.float() / 255.0

        # Pass through the convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten (B, C, H, W) -> (B, N)
        x = x.view(x.size(0), -1)

        # Linear projection to latent space (B, N) -> (B, 512)
        x = self.encoded_features(x)
        return x


# === Predictor Model ===========================================================
class PredictorModel(nn.Module, ABC):
    """Prediction network that maps encoded features back to the latent space.

    Often used as the *online* network in BYOL‑style frameworks. It shares the
    same convolutional stem as *TargetModel* but adds a two‑layer MLP head.
    """

    def __init__(self, state_shape: Tuple[int, int, int]):
        super().__init__()

        # Re‑use the same convolutional backbone definition
        c, w, h = state_shape
        self.conv1 = nn.Conv2d(c,   32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,  64, kernel_size=3, stride=2, padding=1)

        # Flattened output size after conv stack (see comment above)
        conv_out_size = 64

        # Two‑layer MLP head. The second layer uses a much smaller gain during
        # initialisation (0.1) to stabilise training, as recommended in the
        # BYOL paper.
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 512)

        self._init_weights()

    def _init_weights(self) -> None:
        """Layer‑specific orthogonal initialisation.

        * Conv layers use He gain (sqrt(2)).
        * `fc1` uses He gain as well.
        * `fc2` uses a smaller gain (sqrt(0.01)) so that the predictions start
          close to zero – a common stabilisation trick in BYOL‑style training.
        """
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                gain = np.sqrt(0.01) if name == "fc2" else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                layer.bias.data.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predictor forward pass returning a 512‑D vector."""
        x = inputs.float() / 255.0  # Rescale to [0, 1]

        # Convolutional encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and feed through MLP head
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
