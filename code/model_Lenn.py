import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


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

        # Encoder - UNet-like without skip connections
        self.encoder = nn.Sequential(
            # input: 3 x 32 x 32 
            self.conv_downsampler(18, 1),  # output: 18 x 32 x 32
            self.conv_downsampler(18, 2),  # output: 18 x 16 x 16
            self.conv_downsampler(18, 1),  # output: 18 x 16 x 16
            self.conv_downsampler(9, 1),  # output: 9 x 16 x 16
            self.conv_downsampler(9, 2),  # output: 9 x 8 x 8
            nn.Flatten(),  # output: 576
            nn.Linear(9 * 8 * 8, 576),
            nn.ReLU(),
            nn.Linear(576, 288),
            nn.ReLU(),
            nn.Linear(288, self.embedding_size),  # output: latent_size
        )

        # Decoder
        self.decoder = nn.Sequential(
            # input: latent_size
            nn.Linear(self.embedding_size, 288),  # output: 288
            nn.Linear(288, 576),  # output: 576
            nn.Linear(576, 9 * 8 * 8),  # output: 576
            nn.Unflatten(1, (9, 8, 8)),  # output: 9 x 8 x 8
            self.conv_upsampler(9, 2),  # output: 9 x 16 x 16
            self.conv_upsampler(9, 1),  # output: 9 x 16 x 16
            self.conv_upsampler(18, 1),  # output: 18 x 16 x 16
            self.conv_upsampler(18, 2),  # output: 18 x 32 x 32
            self.conv_upsampler(3, 1, use_activation=True),  # output: 3 x 32 x 32
        )

        # Classifier part encapsulated within a Sequential module
        fc1_input_size = self.embedding_size * self.num_bins
        self.mlp_classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(fc1_input_size, self.fc2_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.fc2_size, self.fc2_size // 2),
            nn.ReLU(),
            nn.Linear(self.fc2_size // 2, self.num_classes)
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

    @staticmethod
    def kde(features, num_bins, bandwidth):
        """
        Performs kernel density estimation on a set of features.

        Args:
            features (torch.Tensor): The input tensor containing the features
                                    with shape (batch_size, num_instances, num_features).
            num_bins (int): The number of bins to use for the kernel density estimation.
            bandwidth (float): The bandwidth of the Gaussian kernel.

        Returns:
            torch.Tensor: The output tensor containing the concatenated kernel density
                        estimations for all features across the bins.
        """
        # Get the device from the input features tensor to ensure
        # all operations are on the same device (CPU or GPU).
        device = features.device

        # Extract the dimensions for batch size, number of instances, and feature count.
        batch_size, num_instances, num_features = features.shape

        # Generate equally spaced sample points for KDE in the range [0, 1].
        sample_points = torch.linspace(0, 1, steps=num_bins, device=device)
        sample_points = sample_points.expand(batch_size, num_instances, num_bins)

        # Define the normalization constant and the exponent coefficient for the Gaussian kernel.
        normalization_const = 1 / np.sqrt(2 * np.pi * bandwidth ** 2)
        exponent_coeff = -1 / (2 * bandwidth ** 2)

        # Loop through each feature to compute its KDE.
        feature_kde_list = []
        for feature_index in range(num_features):
            # Select the current feature across all instances and expand it for KDE computation.
            # Compute the squared difference between sample points and the current feature.
            # Apply the Gaussian kernel formula to compute the density estimation.
            # Sum the density estimations across instances for normalization.
            # Normalize the density estimation to ensure it integrates to one.

            current_feature = features[:, :, feature_index].unsqueeze(2).expand_as(sample_points)
            squared_diff = (sample_points - current_feature) ** 2
            density_estimation = normalization_const * torch.exp(exponent_coeff * squared_diff)
            density_sum = density_estimation.sum(dim=1, keepdim=True)
            normalized_density = density_estimation / density_sum.sum(dim=2,
                                                                      keepdim=True)  # Ensure the densities integrate to one across bins.

            feature_kde_list.append(normalized_density)

        # Concatenate the KDE results for all features along the last dimension.
        concatenated_kde = torch.cat(feature_kde_list, dim=-1)

        return concatenated_kde

    def forward(self, x, label=None):
        # Shape Bag
        batch_size, num_instances, num_channel, _, _ = x.shape
        x = x.view(-1, num_channel, x.shape[-2], x.shape[-1])

        # Generate Features (vector embeddings)
        embedding = self.encoder(x)

        # Head1: AutoEncoder
        decoded_img = self.decoder(embedding)

        # Head2: KDE
        embeddings_reshaped = embedding.view(batch_size, num_instances, embedding.shape[-1])
        feature_distribution = self.kde(embeddings_reshaped, self.num_bins, self.sigma)
        logits = self.mlp_classifier(feature_distribution)

        # If labels are provided, compute the loss as a combination of UCC and autoencoder losses
        # TODO refactor this to the training loop
        if label is not None:
            ucc_loss = F.cross_entropy(logits, label)  # Use logits here for numerical stability
            ae_loss = F.mse_loss(decoded_img, x)
            return 0.5 * ucc_loss + 0.5 * ae_loss

        # If no labels are provided, return the logits and reconstructed input
        return logits, decoded_img


if __name__ == "__main__":
    print("############################\\n#Testing Outputs\n############################\n")
    # Define the model (assuming UCCModel class is already defined)
    model = UCCModel(12, 0.5, 0.1, 5, 110, 512)

    # Generate a batch of random data
    batch_size, num_instances, channels, height, width = 2, 5, 3, 32, 32
    random_data = torch.randn((batch_size, num_instances, channels, height, width))

    # Forward pass through the model
    logits, decoded_imgs = model(random_data)

    # Verify output shapes
    print("Random Data::", random_data.shape)
    print("Logits shape:", logits.shape)
    print("Decoded images shape:", decoded_imgs.shape)

    # Visualization
    # Convert the decoded images to a grid for easier visualization
    decoded_img_grid = make_grid(decoded_imgs[0], nrow=num_instances)
    plt.imshow(decoded_img_grid.permute(1, 2, 0).detach().cpu().numpy())
    plt.title("Reconstructed Images")
    plt.axis('off')
    plt.show()

    """
    Random Data:: torch.Size([2, 5, 3, 32, 32])
    Logits shape: torch.Size([2, 5, 5])
    Decoded images shape: torch.Size([10, 3, 32, 32])
    """
