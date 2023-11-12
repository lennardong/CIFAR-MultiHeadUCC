import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_bins: int = 20
    sigma: float = 0.3
    dropout_rate: float = 0.1
    num_classes: int = 4
    embedding_size: int = 200
    fc2_size: int = 512


class UCCModel(nn.Module):
    def __init__(self, num_bins, sigma, dropout_rate, num_classes, embedding_size, fc2_size):
        super(UCCModel, self).__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.fc2_size = fc2_size

        # State Variables
        self.prev_out_channels = 3  # initialize w 3 (3x32x32)

        # Encoder: Simplified U-Net style without skip connections
        self.encoder = nn.Sequential(
            self.conv_downsampler(16, 1),  # output: 16 x 32 x 32
            self.conv_downsampler(16, 1),  # output: 16 x 32 x 32 (xtra)
            self.conv_downsampler(32, 2),  # output: 32 x 16 x 16
            self.conv_downsampler(32, 1),  # output: 32 x 16 x 16 (xtra)
            self.conv_downsampler(64, 2),  # output: 64 x 8 x 8
        )

        # KDE Embeddings: Flatten and sigmoid activation
        self.kde_embeddings = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the encoder
            nn.Linear(64 * 8 * 8, self.embedding_size * 2),  # 4096
            nn.ReLU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.Sigmoid()
        )

        # Decoder: Simplified U-Net style without skip connections
        self.decoder = nn.Sequential(
            self.conv_upsampler(64, 2),  # output: 64 x 16 x 16
            self.conv_upsampler(64, 1),  # output: 64 x 16 x 16 (xtra)
            self.conv_upsampler(32, 2),  # output: 32 x 32 x 32
            self.conv_upsampler(32, 1),  # output: 32 x 32 x 32 (xtra)
            self.conv_upsampler(16, 1),  # output: 16 x 32 x 32
            nn.Conv2d(16, 3, kernel_size=1),  # output: 3 x 32 x 32
        )

        # Classifier part encapsulated within a Sequential module
        self.fc1_input_size = self.embedding_size * self.num_bins
        self.mlp_classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.fc1_input_size, self.fc2_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.fc2_size, self.fc2_size // 2),
            nn.ReLU(),
            nn.Linear(self.fc2_size // 2, self.num_classes),
        )

        # Initialize the weights and biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def conv_downsampler(self, out_channels: int, downsample_factor: int, ks=3, use_activation=True) -> nn.Module:
        """ Return a CNN layer that is same sized or downsampled with optional ReLU activation
        """
        conv_layer = nn.Conv2d(
            in_channels=self.prev_out_channels,  # use prev layer as input to this layer
            out_channels=out_channels,
            kernel_size=ks,
            stride=downsample_factor,
            padding=ks // 2
        )
        activation_layer = nn.ReLU(inplace=True) if use_activation else nn.Identity()

        self.prev_out_channels = out_channels  # Update the last output channels

        return nn.Sequential(*[conv_layer, activation_layer])

    def conv_upsampler(self, out_channels: int, upsample_factor: int, ks=3, use_activation=True) -> nn.Module:
        """ Return a CNN layer that is same-sized or upsampled with optional ReLU activation
        """
        padding = ks // 2  # Adjust padding here
        output_padding = 0  # Initialize output_padding

        if upsample_factor > 1:
            output_padding = upsample_factor - 1  # This will add one cell to each side of the output feature map

        conv_layer = nn.ConvTranspose2d(
            in_channels=self.prev_out_channels,  # use prev layer as input to this layer
            out_channels=out_channels,
            kernel_size=ks,
            stride=upsample_factor,
            padding=padding,
            output_padding=output_padding  # Add output_padding
        )
        activation_layer = nn.ReLU(inplace=True) if use_activation else nn.Identity()

        self.prev_out_channels = out_channels  # Update the last output channels

        return nn.Sequential(*[conv_layer, activation_layer])

    def forward(self, x):
        # get channel shapes
        batch_size, num_instances, num_channel, height, width = x.shape
        x_flat = x.view(-1, num_channel, height, width)

        # Encoder
        embeddings_conv = self.encoder(x_flat)

        # Head1: Decoder
        decoded_img_flat = self.decoder(embeddings_conv)
        decoded_img = decoded_img_flat.view(batch_size, num_instances, num_channel, height, width)  # Shape: (batch_size, num_instances, num_channel, height, width)

        # Head2: KDE
        embeddings_fc = self.kde_embeddings(embeddings_conv)
        embeddings_reshaped = embeddings_fc.view(batch_size, num_instances, embeddings_fc.shape[-1])  # Shape: (batch_size, num_instances, embedding_size)
        feature_distribution = self.kde(embeddings_reshaped, self.num_bins,
                                                  self.sigma)  # Shape: (batch_size, num_bins * embedding_size)

        logits = self.mlp_classifier(feature_distribution)

        return logits, decoded_img

    @staticmethod
    def loss_function_multihead(logits, decoded_img, labels, original_imgs, ucc_loss_weight=0.5):

        ae_loss_weight = 1 - ucc_loss_weight

        ucc_loss = F.cross_entropy(logits, labels)
        ae_loss = F.mse_loss(decoded_img, original_imgs)
        combined_loss = (ucc_loss_weight * ucc_loss) + (ae_loss_weight * ae_loss)

        return ucc_loss, ae_loss, combined_loss

    @staticmethod
    def kde(data, num_bins, bandwidth):

        # Inits
        device = data.device
        batch_size, num_instances, num_features = data.shape

        # Sampling
        sample_points = torch.linspace(0, 1, steps=num_bins, device=device).expand(batch_size, num_instances, num_bins)

        # Kernal consts
        normalization_const = 1 / (bandwidth * np.sqrt(2 * np.pi))
        exponent_coeff = -1 / (2 * bandwidth ** 2)

        # Compute
        kde_results = []
        for feature_index in range(num_features):

            # Features across all instances & expand it to match the number of bins.
            # Expected shape: (batch_size, num_instances, num_bins)
            current_feature = data[:, :, feature_index:feature_index + 1]  # Shape: (batch_size, num_instances, 1)
            current_feature = current_feature.expand(-1, -1, num_bins)  # Shape: (batch_size, num_instances, num_bins)

            # Squared differences between sample points and the current feature.
            # Expected shape: (batch_size, num_instances, num_bins)
            squared_diff = (sample_points - current_feature) ** 2

            # Gaussian kernel to the squared differences.
            # Expected shape: (batch_size, num_instances, num_bins)
            gaussian_kernel = normalization_const * torch.exp(exponent_coeff * squared_diff)

            # Integrate densities: sum the Gaussian kernel results across instances
            # Expected shape after sum: (batch_size, num_bins)
            integrated_densities = gaussian_kernel.sum(1)

            # Compute normalization factor for each batch to ensure the densities sum to 1.
            # Expected shape after sum: (batch_size, 1)
            normalization_factors = integrated_densities.sum(1, keepdim=True)

            # Normalize the KDE values for each feature.
            # Expected shape: (batch_size, num_bins)
            normalized_kde = integrated_densities / normalization_factors

            # Collect the KDE results for this feature.
            kde_results.append(normalized_kde)

        # Concatenate the KDE results for all features into a single tensor.
        # Expected final shape: (batch_size, num_bins * num_features)
        concatenated_kde = torch.cat(kde_results, dim=1)

        return concatenated_kde


############################################
# MAIN
############################################

if __name__ == "__main__":
    print("############################\n# Testing Outputs\n############################\n")
    # Define the model (assuming UCCModel class is already defined)
    model = UCCModel(12, 0.5, 0.1, 5, 110, 512)

    # Generate a batch of random data
    batch_size, num_instances, channels, height, width = 2, 5, 3, 32, 32
    random_data = torch.randn((batch_size, num_instances, channels, height, width))

    # Forward pass through the model
    logits, decoded_imgs = model(random_data)

    # Verify output shapes
    print("Random Data:", random_data.shape)
    print("Logits shape:", logits.shape)
    print("Decoded images shape:", decoded_imgs.shape)

    """
    Random Data:: torch.Size([2, 5, 3, 32, 32])
    Logits shape: torch.Size([2, 5, 5])
    Decoded images shape: torch.Size([10, 3, 32, 32])
    """
