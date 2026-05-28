"""Multi-transform combo processing for video perturbation.

This module implements the MultiTransformCombo class which orchestrates
combining multiple perturbation groups (Temporal, Spatial, Audio) with
scaled intensity. It selects 2-3 groups, applies them in fixed order,
and handles graceful degradation if a group fails.
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from typing import Callable

from .perturbation_config import PerturbationConfig

logger = logging.getLogger(__name__)

# Fixed application order: Temporal → Spatial → Audio
APPLICATION_ORDER = ["temporal", "spatial", "audio"]

# Available groups mapped to their enabled config flag
_GROUP_ENABLED_FIELD = {
    "temporal": "temporal_enabled",
    "spatial": "spatial_enabled",
    "audio": "audio_enabled",
}


class MultiTransformCombo:
    """Orchestrates combining multiple perturbation groups.

    Selects 2-3 groups from available perturbation groups, scales their
    intensity by combo_intensity_factor, and applies them in fixed order
    (Temporal → Spatial → Audio). If a group fails during application,
    it is skipped and remaining groups continue.
    """

    def __init__(self, config: PerturbationConfig, rng: random.Random) -> None:
        """Initialize the multi-transform combo.

        Args:
            config: Perturbation configuration with combo parameters.
            rng: Seeded Random instance for reproducibility.
        """
        self.config = config
        self.rng = rng
        self._selected_groups: list[str] | None = None

    @property
    def selected_groups(self) -> list[str]:
        """Get the currently selected groups, selecting if not yet done."""
        if self._selected_groups is None:
            self._selected_groups = self.select_groups()
        return self._selected_groups

    def select_groups(self) -> list[str]:
        """Select 2-3 groups randomly from available groups.

        Only groups that are enabled in the config are available for
        selection. The returned list is sorted in the fixed application
        order (Temporal → Spatial → Audio).

        Returns:
            List of 2-3 group names sorted in application order.
        """
        available = [
            group
            for group, field in _GROUP_ENABLED_FIELD.items()
            if getattr(self.config, field)
        ]

        if len(available) < 2:
            # If fewer than 2 groups available, return all available
            return sorted(available, key=lambda g: APPLICATION_ORDER.index(g))

        # Select 2 or 3 groups randomly
        max_count = min(3, len(available))
        count = self.rng.randint(2, max_count)
        selected = self.rng.sample(available, count)

        # Sort in fixed application order
        selected.sort(key=lambda g: APPLICATION_ORDER.index(g))
        return selected

    def scale_intensity(
        self, base_value: float, param_min: float, param_max: float
    ) -> float:
        """Scale parameter intensity by combo_intensity_factor.

        Applies the formula: result = param_min + (base_value - param_min) * factor
        This reduces the deviation from minimum, effectively reducing intensity
        when factor < 1.0.

        The result is clamped to [param_min, param_max].

        Args:
            base_value: The standalone parameter value to scale.
            param_min: Minimum of the parameter range.
            param_max: Maximum of the parameter range.

        Returns:
            Scaled parameter value within [param_min, param_max].
        """
        factor = self.config.combo_intensity_factor
        result = param_min + (base_value - param_min) * factor
        # Clamp to valid range
        return max(param_min, min(result, param_max))

    def create_scaled_config(self) -> PerturbationConfig:
        """Create a new config with intensity-scaled parameters for selected groups.

        Returns a modified PerturbationConfig where parameters for the
        selected groups are scaled by combo_intensity_factor. Only parameters
        that represent intensity/deviation are scaled.

        Returns:
            New PerturbationConfig with scaled parameters.
        """
        groups = self.selected_groups
        overrides: dict[str, float] = {}

        if "temporal" in groups:
            # Scale speed deviation from 1.0
            # speed_min is in [0.90, 1.0], deviation = 1.0 - speed_min
            overrides["speed_min"] = self.scale_intensity(
                self.config.speed_min, 1.0, 0.90
            )
            # For speed_min, param_min=1.0 (no deviation), param_max=0.90
            # Actually, speed_min ranges [0.90, 1.0] where 1.0 = no effect
            # We want to scale toward 1.0 (less deviation)
            # result = 1.0 + (speed_min - 1.0) * factor
            overrides["speed_min"] = 1.0 + (self.config.speed_min - 1.0) * self.config.combo_intensity_factor

            # speed_max is in [1.0, 1.10], where 1.0 = no effect
            # result = 1.0 + (speed_max - 1.0) * factor
            overrides["speed_max"] = 1.0 + (self.config.speed_max - 1.0) * self.config.combo_intensity_factor

            # max_frame_drop_percent: [0, 20], 0 = no effect
            overrides["max_frame_drop_percent"] = self.scale_intensity(
                self.config.max_frame_drop_percent, 0.0, 20.0
            )

            # micro_offset_ms: [0, 500], 0 = no effect
            overrides["micro_offset_ms"] = self.scale_intensity(
                self.config.micro_offset_ms, 0.0, 500.0
            )

        if "spatial" in groups:
            # max_crop_percent: [0, 50], 0 = no effect
            overrides["max_crop_percent"] = self.scale_intensity(
                self.config.max_crop_percent, 0.0, 50.0
            )

            # max_zoom: [1.0, 2.0], 1.0 = no effect
            overrides["max_zoom"] = self.scale_intensity(
                self.config.max_zoom, 1.0, 2.0
            )

        if "audio" in groups:
            # audio_tempo_min: [0.90, 1.0], 1.0 = no effect
            overrides["audio_tempo_min"] = 1.0 + (self.config.audio_tempo_min - 1.0) * self.config.combo_intensity_factor

            # audio_tempo_max: [1.0, 1.10], 1.0 = no effect
            overrides["audio_tempo_max"] = 1.0 + (self.config.audio_tempo_max - 1.0) * self.config.combo_intensity_factor

            # eq_range_db: [0, 12], 0 = no effect
            overrides["eq_range_db"] = self.scale_intensity(
                self.config.eq_range_db, 0.0, 12.0
            )

        return replace(self.config, **overrides)

    def apply_groups(
        self,
        group_handlers: dict[str, Callable[[], None]],
    ) -> list[str]:
        """Apply selected groups in fixed order with graceful degradation.

        Executes the handler for each selected group in the fixed order
        (Temporal → Spatial → Audio). If a group's handler raises an
        exception, it is caught, logged, and the remaining groups continue.

        Args:
            group_handlers: Dict mapping group name to a callable that
                applies that group's perturbation. Each callable takes
                no arguments and returns None.

        Returns:
            List of group names that were successfully applied.
        """
        groups = self.selected_groups
        applied: list[str] = []

        for group in APPLICATION_ORDER:
            if group not in groups:
                continue

            handler = group_handlers.get(group)
            if handler is None:
                logger.warning(
                    "No handler registered for group '%s', skipping.", group
                )
                continue

            try:
                handler()
                applied.append(group)
            except Exception as exc:
                logger.warning(
                    "Group '%s' failed during combo application: %s. "
                    "Skipping and continuing with remaining groups.",
                    group,
                    exc,
                )

        return applied
