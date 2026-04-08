# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

import enum
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Boolean
from cutlass.cutlass_dsl import (
    Float32,
    Int32,
    extract_mlir_values,
    min,
    new_from_mlir_values,
)
from cutlass.utils import WorkTileInfo
from cutlass.utils.hardware_info import HardwareInfo


class FmhaStaticTileSchedulerParams:
    """Parameters for the FMHA static tile scheduler."""

    def __init__(
        self,
        is_persistent: bool,
        problem_shape_mbh: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.is_persistent = is_persistent
        self.problem_shape_mbh = problem_shape_mbh
        self._loc = loc
        self._ip = ip

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.problem_shape_mbh]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.problem_shape_mbh], self._values_pos):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FmhaStaticTileSchedulerParams(
            self.is_persistent, *(tuple(obj_list)), loc=self._loc
        )


class FmhaStaticTileScheduler:
    """Static tile scheduler used by the Hopper FMHA example."""

    def __init__(
        self,
        params: FmhaStaticTileSchedulerParams,
        current_work_linear_idx: Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self._params = params
        self._blk_coord = blk_coord
        self._grid_shape = grid_shape
        self._is_persistent = params.is_persistent
        self._current_work_linear_idx = current_work_linear_idx
        self._problem_shape_mbh = cute.make_layout(
            params.problem_shape_mbh, loc=loc, ip=ip
        )
        self._num_blocks = cute.size(self._problem_shape_mbh, loc=loc, ip=ip)
        self._is_first_block = True
        self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        self._loc = loc
        self._ip = ip

    @staticmethod
    def get_grid_shape(
        params: FmhaStaticTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        if params.is_persistent:
            hardware_info = HardwareInfo()
            sm_count = hardware_info.get_device_multiprocessor_count()
            return (
                min(sm_count, cute.size(params.problem_shape_mbh, loc=loc, ip=ip)),
                1,
                1,
            )
        return params.problem_shape_mbh

    @staticmethod
    def check_valid_work_for_seqlen_q(
        q_tiler: int,
        current_idx: Int32,
        seqlen_q: Int32,
    ) -> Boolean:
        return current_idx * q_tiler < seqlen_q

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        is_valid = (
            self._current_work_linear_idx < self._num_blocks
            if self._is_persistent
            else self._is_first_block
        )

        blk_coord = (0, 0, 0)
        if self._is_persistent:
            blk_coord = self._problem_shape_mbh.get_hier_coord(
                self._current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            blk_coord = self._blk_coord

        cur_tile_coord = (
            blk_coord[0],
            0,
            (blk_coord[1], blk_coord[2]),
        )
        return WorkTileInfo(cur_tile_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self._is_persistent:
            self._current_work_linear_idx += advance_count * self.num_persistent_sm
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self._params)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self._blk_coord))
        values.extend(extract_mlir_values(self._grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 10
        new_params = new_from_mlir_values(self._params, values[0:3])
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[3]]
        )
        new_blk_coord = new_from_mlir_values(self._blk_coord, values[4:7])
        new_grid_shape = new_from_mlir_values(self._grid_shape, values[7:])
        return FmhaStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


def create_fmha_static_tile_scheduler(
    params: FmhaStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> FmhaStaticTileScheduler:
    return FmhaStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


def create_fmha_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_mbh: cute.Shape,
) -> FmhaStaticTileSchedulerParams:
    return FmhaStaticTileSchedulerParams(is_persistent, problem_shape_mbh)


def compute_grid(
    o_shape: cute.Shape,
    cta_tiler: Tuple[int, int, int],
    is_persistent: bool,
) -> Tuple[FmhaStaticTileSchedulerParams, Tuple[int, int, int]]:
    tile_sched_params = create_fmha_static_tile_scheduler_params(
        is_persistent,
        (
            cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]),
            cute.size(o_shape[2][0]),
            cute.size(o_shape[2][1]),
        ),
    )
    grid = FmhaStaticTileScheduler.get_grid_shape(tile_sched_params)
    return tile_sched_params, grid


class MaskEnum(enum.Enum):
    RESIDUAL_MASK = enum.auto()
    RESIDUAL_MASK_BWD = enum.auto()
    WINDOW_MASK = enum.auto()
    WINDOW_MASK_INFERENCE = enum.auto()
    WINDOW_MASK_BWD = enum.auto()
    WINDOW_MASK_BWD_INFERENCE = enum.auto()


class FusedMask:
    def get_trip_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(mask_type == MaskEnum.RESIDUAL_MASK):
            result = cute.ceil_div(seqlen_k, tile_shape[1])
        if cutlass.const_expr(mask_type is MaskEnum.RESIDUAL_MASK_BWD):
            result = cute.ceil_div(seqlen_q, tile_shape[0])
        if cutlass.const_expr(
            mask_type == MaskEnum.WINDOW_MASK
            or mask_type == MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is None):
                result = cute.ceil_div(seqlen_k, tile_shape[1])
            else:
                max_idx_q = (blk_coord[0] + 1) * tile_shape[0]
                idx_k = max_idx_q + offset + window_size_right
                tmp_blocks_k = cute.ceil_div(idx_k, tile_shape[1])
                max_blocks_k = cute.ceil_div(seqlen_k, tile_shape[1])
                result = min(max_blocks_k, tmp_blocks_k)
        if cutlass.const_expr(
            mask_type == MaskEnum.WINDOW_MASK_BWD
            or mask_type == MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is None):
                result = cute.ceil_div(seqlen_q, tile_shape[0])
            else:
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1]
                idx_k = max_idx_k + offset + window_size_left
                tmp_blocks_q = cute.ceil_div(idx_k, tile_shape[0])
                max_blocks_q = cute.ceil_div(seqlen_q, tile_shape[0])
                result = min(max_blocks_q, tmp_blocks_q)
        start_block = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        result = result - start_block
        return result

    @cute.jit
    def get_trip_start(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK
            or mask_type is MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = blk_coord[0] * tile_shape[0]
                idx_k = min_idx_q + offset - window_size_left
                tmp_blocks_k = idx_k // tile_shape[1]
                result = max(tmp_blocks_k, result)
        if cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK_BWD
            or mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = blk_coord[1] * tile_shape[1]
                idx_q = min_idx_k + offset - window_size_right
                tmp_blocks_q = idx_q // tile_shape[0]
                result = max(tmp_blocks_q, result)
        return result

    @cute.jit
    def get_leading_mask_id(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        leading_mask_begin = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        leading_mask_end = leading_mask_begin
        if cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK
            or mask_type is MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = (
                    (blk_coord[0] + 1) * tile_shape[0] + offset - window_size_left
                )
                leading_mask_end = min(
                    cute.ceil_div(min_idx_q, tile_shape[1]) - 1,
                    trip_count + leading_mask_begin - 1,
                )
            else:
                leading_mask_end = leading_mask_begin - 1
        elif cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK_BWD
            or mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = (
                    (blk_coord[1] + 1) * tile_shape[1] + offset - window_size_right
                )
                leading_mask_end = cute.ceil_div(min_idx_k, tile_shape[0]) - 1
            else:
                leading_mask_end = leading_mask_begin - 1
        return leading_mask_begin, leading_mask_end

    @cute.jit
    def get_trailing_mask_id(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Optional[Int32], Optional[Int32]]:
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        trip_start = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        trailing_mask_begin, trailing_mask_end = None, None
        if cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK
            or mask_type is MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_q = blk_coord[0] * tile_shape[0] + offset + window_size_right
                trailing_mask_begin = min(
                    min_idx_q // tile_shape[1], trip_count + trip_start - 1
                )
                trailing_mask_end = trip_count + trip_start - 1
            else:
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1
        else:
            if cutlass.const_expr(window_size_left is not None):
                min_idx_k = blk_coord[1] * tile_shape[1] + offset + window_size_left + 1
                max_idx_k = (
                    (blk_coord[1] + 1) * tile_shape[1] + offset + window_size_left
                )
                trailing_mask_begin = min(
                    cute.ceil_div(min_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
                trailing_mask_end = min(
                    cute.ceil_div(max_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
            else:
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1

        return trailing_mask_begin, trailing_mask_end

    @cute.jit
    def get_masked_leading_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        result = 0
        if cutlass.const_expr(
            mask_type is not MaskEnum.RESIDUAL_MASK
            and mask_type is not MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(
                window_size_left is not None or window_size_right is not None
            ):
                leading_mask_begin, leading_mask_end = FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                result = max(leading_mask_end - leading_mask_begin + 1, 0)

        return result

    @cute.jit
    def get_masked_trailing_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        rem_count: Optional[Int32] = 0,
    ) -> Int32:
        result = 0

        if cutlass.const_expr(
            mask_type is not MaskEnum.RESIDUAL_MASK
            and mask_type is not MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(
                window_size_left is not None or window_size_right is not None
            ):
                trailing_mask_begin, trailing_mask_end = FusedMask.get_trailing_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                leading_mask_begin, leading_mask_end = FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                if cutlass.const_expr(
                    trailing_mask_begin is not None and trailing_mask_end is not None
                ):
                    if trailing_mask_begin <= leading_mask_end:
                        result = max(trailing_mask_end - leading_mask_end, 0)
                    else:
                        result = max(trailing_mask_end - trailing_mask_begin + 1, 0)
        else:
            if seqlen_k % tile_shape[1] != 0:
                result = 1
            else:
                result = 0

        return result + rem_count

    @cute.jit
    def get_unmasked_trip_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        result = (
            FusedMask.get_trip_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - FusedMask.get_masked_leading_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - FusedMask.get_masked_trailing_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                0,
            )
        )
        return result

    @cute.jit
    def apply_mask(
        mask_type: MaskEnum,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        tidx, tidy, tidx = cute.arch.thread_idx()
        offset = 0
        offset = (
            seqlen_k - seqlen_q
            if cutlass.const_expr(
                mask_type is MaskEnum.WINDOW_MASK_INFERENCE
                or mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE
            )
            else 0
        )
        for i in cutlass.range_constexpr(cute.size(acc_qk)):
            index_q, index_k = index_transform(*index_qk[i])
            if cutlass.const_expr(
                window_size_left is not None or window_size_right is not None
            ):
                if cutlass.const_expr(window_size_left is None):
                    if index_q + offset + window_size_right < index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:
                        acc_qk[i] = -Float32.inf
                elif cutlass.const_expr(window_size_right is None):
                    if index_q + offset - window_size_left > index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:
                        acc_qk[i] = -Float32.inf
                else:
                    max_K_index = min(index_q + offset + window_size_right, seqlen_k)
                    min_K_index = max(0, index_q + offset - window_size_left)
                    if index_k > max_K_index or index_k < min_K_index:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:
                        acc_qk[i] = -Float32.inf

            if cutlass.const_expr(
                mask_type == MaskEnum.RESIDUAL_MASK
                or mask_type == MaskEnum.RESIDUAL_MASK_BWD
            ):
                if index_k >= seqlen_k or index_q >= seqlen_q:
                    acc_qk[i] = -Float32.inf
