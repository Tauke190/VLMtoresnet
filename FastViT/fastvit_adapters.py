import torch
import torch.nn as nn
import fastvit as fv


class ConvAdapter(nn.Module):

    def __init__(
        self,
        dim: int,
        reduction_ratio: int = 4,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        hidden_dim = max(dim // reduction_ratio, 1)

        self.encoder = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_layer(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)
        y = self.decoder(y)
        return self.scale * y


class RepMixerBlockAdapter(fv.RepMixerBlock):

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
        use_adapter: bool = True,
        adapter_reduction_ratio: int = 4,
    ):
        super().__init__(
            dim=dim,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            drop=drop,
            drop_path=drop_path,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        self.use_adapter = use_adapter
        if self.use_adapter:
            self.adapter = ConvAdapter(
                dim=dim,
                reduction_ratio=adapter_reduction_ratio,
                act_layer=act_layer,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.token_mixer(x)

        ff_in = x
        ff_out = self.convffn(ff_in)

        if self.use_adapter:
            ff_out = ff_out + self.adapter(ff_in)

        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale * ff_out)
        else:
            x = x + self.drop_path(ff_out)

        return x


class AttentionBlockAdapter(fv.AttentionBlock):

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        use_adapter: bool = True,
        adapter_reduction_ratio: int = 4,
    ):
        super().__init__(
            dim=dim,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=drop,
            drop_path=drop_path,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.use_adapter = use_adapter
        if self.use_adapter:
            self.adapter = ConvAdapter(
                dim=dim,
                reduction_ratio=adapter_reduction_ratio,
                act_layer=act_layer,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))

        ff_in = x
        ff_out = self.convffn(ff_in)

        if self.use_adapter:
            ff_out = ff_out + self.adapter(ff_in)

        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * ff_out)
        else:
            x = x + self.drop_path(ff_out)
        return x


fv.RepMixerBlock = RepMixerBlockAdapter
fv.AttentionBlock = AttentionBlockAdapter
