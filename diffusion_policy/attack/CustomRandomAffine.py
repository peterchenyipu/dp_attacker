from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample, SamplePadding
from kornia.core import Tensor, as_tensor
from kornia.geometry.conversions import deg2rad
from kornia.geometry.transform import get_affine_matrix2d, warp_affine



from typing import Dict, Optional, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check, _range_bound
from kornia.core import Tensor, as_tensor, concatenate, stack, tensor, zeros
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["AffineGenerator"]


class CustomAffineGenerator(RandomGeneratorBase):
    r"""Get parameters for ``affine`` for a random affine transform.

    Args:
        degrees: Range of degrees to select from like (min, max).
        translate: tuple of fraction for horizontal and vertical translations.
            If translate=(a, b), then horizontal shift is randomly sampled in the range 
            -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. 
            If translate=(a, b, c, d), then horizontal shift is randomly sampled in the range
            -img_width * a < dx < img_width * b and vertical shift is randomly sampled in the range
            -img_height * c < dy < img_height * d. Will not translate by default.
        scale: scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear: Range of degrees to select from.
            If float, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b), a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b, c, d), then x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3])
            will be applied. Will not apply shear by default.
            If tensor, shear is a 2x2 tensor, a x-axis shear in (shear[0][0], shear[0][1]) and y-axis shear in
            (shear[1][0], shear[1][1]) will be applied. Will not apply shear by default.

    Returns:
        A dict of parameters to be passed for transformation.
            - translations (Tensor): element-wise translations with a shape of (B, 2).
            - center (Tensor): element-wise center with a shape of (B, 2).
            - scale (Tensor): element-wise scales with a shape of (B, 2).
            - angle (Tensor): element-wise rotation angles with a shape of (B,).
            - shear_x (Tensor): element-wise x-axis shears with a shape of (B,).
            - shear_y (Tensor): element-wise y-axis shears with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        degrees: Union[Tensor, float, Tuple[float, float]],
        translate: Optional[Union[Tensor, Tuple[float, float]]] = None,
        scale: Optional[Union[Tensor, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        shear: Optional[Union[Tensor, float, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __repr__(self) -> str:
        repr = f"degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        _degrees = _range_bound(self.degrees, "degrees", 0, (-360, 360)).to(device=device, dtype=dtype)
        _translate: Optional[Tensor] = None
        if self.translate is not None:
            if len(self.translate) == 2:
                _translate = _range_bound(self.translate, "translate", bounds=(0, 1), check="singular").to(
                    device=device, dtype=dtype
                )
            elif len(self.translate) == 4:
                _translate = concatenate(
                    [
                        _range_bound(self.translate[:2], "translate_x", bounds=(-1, 1), check="singular"),
                        _range_bound(self.translate[-2:], "translate_y", bounds=(-1, 1), check="singular"),
                    ]
                ).to(device=device, dtype=dtype)
            else:
                raise ValueError(f"'translate' expected to be either 2 or 4 elements. Got {self.translate}")
        # _translate = (
        #     self.translate
        #     if self.translate is None
        #     else _range_bound(self.translate, "translate", bounds=(0, 1), check="singular").to(
        #         device=device, dtype=dtype
        #     )
        # )
        _scale: Optional[Tensor] = None
        if self.scale is not None:
            if len(self.scale) == 2:
                _scale = _range_bound(self.scale[:2], "scale", bounds=(0, float("inf")), check="singular").to(
                    device=device, dtype=dtype
                )
            elif len(self.scale) == 4:
                _scale = concatenate(
                    [
                        _range_bound(self.scale[:2], "scale_x", bounds=(0, float("inf")), check="singular"),
                        _range_bound(self.scale[-2:], "scale_y", bounds=(0, float("inf")), check="singular"),
                    ]
                ).to(device=device, dtype=dtype)
            else:
                raise ValueError(f"'scale' expected to be either 2 or 4 elements. Got {self.scale}")
        _shear: Optional[Tensor] = None
        if self.shear is not None:
            shear = as_tensor(self.shear, device=device, dtype=dtype)
            if shear.shape == torch.Size([2, 2]):
                _shear = shear
            else:
                _shear = stack(
                    [
                        _range_bound(shear if shear.dim() == 0 else shear[:2], "shear-x", 0, (-360, 360)),
                        (
                            tensor([0, 0], device=device, dtype=dtype)
                            if shear.dim() == 0 or len(shear) == 2
                            else _range_bound(shear[2:], "shear-y", 0, (-360, 360))
                        ),
                    ]
                )

        translate_x_sampler: Optional[UniformDistribution] = None
        translate_y_sampler: Optional[UniformDistribution] = None
        scale_2_sampler: Optional[UniformDistribution] = None
        scale_4_sampler: Optional[UniformDistribution] = None
        shear_x_sampler: Optional[UniformDistribution] = None
        shear_y_sampler: Optional[UniformDistribution] = None

        if _translate is not None:
            # translate_x_sampler = UniformDistribution(-_translate[0], _translate[0], validate_args=False)
            # translate_y_sampler = UniformDistribution(-_translate[1], _translate[1], validate_args=False)
            # print(_translate)
            if len(_translate) == 2:
                translate_x_sampler = UniformDistribution(-_translate[0], _translate[0], validate_args=False)
                translate_y_sampler = UniformDistribution(-_translate[1], _translate[1], validate_args=False)
            elif len(_translate) == 4:
                translate_x_sampler = UniformDistribution(_translate[0], _translate[1], validate_args=False)
                translate_y_sampler = UniformDistribution(_translate[2], _translate[3], validate_args=False)
        if _scale is not None:
            if len(_scale) == 2:
                scale_2_sampler = UniformDistribution(_scale[0], _scale[1], validate_args=False)
            elif len(_scale) == 4:
                scale_2_sampler = UniformDistribution(_scale[0], _scale[1], validate_args=False)
                scale_4_sampler = UniformDistribution(_scale[2], _scale[3], validate_args=False)
            else:
                raise ValueError(f"'scale' expected to be either 2 or 4 elements. Got {self.scale}")
        if _shear is not None:
            _joint_range_check(_shear[0], "shear")
            _joint_range_check(_shear[1], "shear")
            shear_x_sampler = UniformDistribution(_shear[0][0], _shear[0][1], validate_args=False)
            shear_y_sampler = UniformDistribution(_shear[1][0], _shear[1][1], validate_args=False)

        self.degree_sampler = UniformDistribution(_degrees[0], _degrees[1], validate_args=False)
        self.translate_x_sampler = translate_x_sampler
        self.translate_y_sampler = translate_y_sampler
        self.scale_2_sampler = scale_2_sampler
        self.scale_4_sampler = scale_4_sampler
        self.shear_x_sampler = shear_x_sampler
        self.shear_y_sampler = shear_y_sampler

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.degrees, self.translate, self.scale, self.shear])
        _common_param_check(batch_size, same_on_batch)
        if not (isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0):
            raise AssertionError(f"`width` and `height` must be positive integers. Got {width}, {height}.")

        angle = _adapted_rsampling((batch_size,), self.degree_sampler, same_on_batch).to(device=_device, dtype=_dtype)

        # compute tensor ranges
        if self.scale_2_sampler is not None:
            _scale = _adapted_rsampling((batch_size,), self.scale_2_sampler, same_on_batch).unsqueeze(1).repeat(1, 2)
            if self.scale_4_sampler is not None:
                _scale[:, 1] = _adapted_rsampling((batch_size,), self.scale_4_sampler, same_on_batch)
            _scale = _scale.to(device=_device, dtype=_dtype)
        else:
            _scale = torch.ones((batch_size, 2), device=_device, dtype=_dtype)

        if self.translate_x_sampler is not None and self.translate_y_sampler is not None:
            translations = stack(
                [
                    _adapted_rsampling((batch_size,), self.translate_x_sampler, same_on_batch) * width,
                    _adapted_rsampling((batch_size,), self.translate_y_sampler, same_on_batch) * height,
                ],
                dim=-1,
            )
            translations = translations.to(device=_device, dtype=_dtype)
        else:
            translations = zeros((batch_size, 2), device=_device, dtype=_dtype)

        center: Tensor = tensor([width, height], device=_device, dtype=_dtype).view(1, 2) / 2.0 - 0.5
        center = center.expand(batch_size, -1)

        if self.shear_x_sampler is not None and self.shear_y_sampler is not None:
            sx = _adapted_rsampling((batch_size,), self.shear_x_sampler, same_on_batch)
            sy = _adapted_rsampling((batch_size,), self.shear_y_sampler, same_on_batch)
            sx = sx.to(device=_device, dtype=_dtype)
            sy = sy.to(device=_device, dtype=_dtype)
        else:
            sx = tensor([0] * batch_size, device=_device, dtype=_dtype)
            sy = tensor([0] * batch_size, device=_device, dtype=_dtype)

        # print(f'{translations=}')

        return {
            "translations": translations,
            "center": center,
            "scale": _scale,
            "angle": angle,
            "shear_x": sx,
            "shear_y": sy,
        }




class CustomRandomAffine(GeometricAugmentationBase2D):
    r"""Apply a random 2D affine transformation to a tensor image.

    .. image:: _static/img/RandomAffine.png

    The transformation is computed so that the image center is kept invariant.

    Args:
        degrees: Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate: tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale: scaling factor interval.
            If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b.
            If (a, b, c, d), the scale is randomly sampled from the range a <= scale_x <= b, c <= scale_y <= d.
            Will keep original scale by default.
        shear: Range of degrees to select from.
            If float, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b), a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b, c, d), then x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3])
            will be applied. Will not apply shear by default.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        padding_mode: padding mode from "zeros" (0), "border" (1) or "reflection" (2).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_affine`.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3)
        >>> aug = RandomAffine((-15., 20.), p=1.)
        >>> out = aug(input)
        >>> out, aug.transform_matrix
        (tensor([[[[0.3961, 0.7310, 0.1574],
                  [0.1781, 0.3074, 0.5648],
                  [0.4804, 0.8379, 0.4234]]]]), tensor([[[ 0.9923, -0.1241,  0.1319],
                 [ 0.1241,  0.9923, -0.1164],
                 [ 0.0000,  0.0000,  1.0000]]]))
        >>> aug.inverse(out)
        tensor([[[[0.3890, 0.6573, 0.1865],
                  [0.2063, 0.3074, 0.5459],
                  [0.3892, 0.7896, 0.4224]]]])
        >>> input
        tensor([[[[0.4963, 0.7682, 0.0885],
                  [0.1320, 0.3074, 0.6341],
                  [0.4901, 0.8964, 0.4556]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomAffine((-15., 20.), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        degrees: Union[Tensor, float, Tuple[float, float]],
        translate: Optional[Union[Tensor, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        scale: Optional[Union[Tensor, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        shear: Optional[Union[Tensor, float, Tuple[float, float]]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator: rg.AffineGenerator = CustomAffineGenerator(degrees, translate, scale, shear)
        self.flags = {
            "resample": Resample.get(resample),
            "padding_mode": SamplePadding.get(padding_mode),
            "align_corners": align_corners,
        }

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return get_affine_matrix2d(
            as_tensor(params["translations"], device=input.device, dtype=input.dtype),
            as_tensor(params["center"], device=input.device, dtype=input.dtype),
            as_tensor(params["scale"], device=input.device, dtype=input.dtype),
            as_tensor(params["angle"], device=input.device, dtype=input.dtype),
            deg2rad(as_tensor(params["shear_x"], device=input.device, dtype=input.dtype)),
            deg2rad(as_tensor(params["shear_y"], device=input.device, dtype=input.dtype)),
        )

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, _, height, width = input.shape
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")

        return warp_affine(
            input,
            transform[:, :2, :],
            (height, width),
            flags["resample"].name.lower(),
            align_corners=flags["align_corners"],
            padding_mode=flags["padding_mode"].name.lower(),
        )

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")
        return self.apply_transform(
            input,
            params=self._params,
            transform=as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
