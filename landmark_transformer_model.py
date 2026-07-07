import torch
import torch.nn as nn


class LandmarkTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 3,
        num_landmarks: int = 21,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_landmarks + 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x shape:
        - (batch, 63)
        hoặc
        - (batch, 21, 3)
        """

        if x.dim() == 2:
            x = x.view(-1, self.num_landmarks, self.input_dim)

        batch_size = x.size(0)

        x = self.input_projection(x)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1,
        )

        x = x + self.position_embedding

        x = self.encoder(x)

        cls_output = x[:, 0]

        cls_output = self.norm(cls_output)

        logits = self.classifier(cls_output)

        return logits