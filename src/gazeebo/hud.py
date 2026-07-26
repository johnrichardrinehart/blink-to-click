"""Opt-in pointer diagnostics on a transparent Wayland layer surface."""

from __future__ import annotations

import ctypes
import time
from typing import TYPE_CHECKING, Protocol

from gazeebo.native import NativeRendererError, load_native_renderer
from gazeebo.portal import PortalError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gazeebo.contracts import DisplayRegion, TrainingRegionStatus

HUD_INTERVAL_SECONDS = 1.0
HUD_ERROR_SIZE = 256


class _Surface(Protocol):
    """Graphical operations needed by the policy-free HUD limiter."""

    def show(self, region_id: str, x: float, y: float) -> None:
        """Display the latest output and coordinate diagnostics."""

    def close(self) -> None:
        """Destroy the surface."""


class _NativeLayerSurface:
    """Load the in-process Wayland shared-memory renderer."""

    def __init__(self) -> None:
        """Create one layer surface through the packaged native library."""
        try:
            library = load_native_renderer()
        except NativeRendererError as load_error:
            raise PortalError(str(load_error)) from load_error
        library.gazeebo_hud_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        library.gazeebo_hud_create.restype = ctypes.c_void_p
        library.gazeebo_hud_update.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_hud_update.restype = ctypes.c_int
        library.gazeebo_hud_destroy.argtypes = [ctypes.c_void_p]
        library.gazeebo_hud_destroy.restype = None
        error_buffer = ctypes.create_string_buffer(HUD_ERROR_SIZE)
        handle = library.gazeebo_hud_create(error_buffer, HUD_ERROR_SIZE)
        if not handle:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"debug HUD failed to start: {detail}"
            raise PortalError(msg)
        self._library = library
        self._handle = ctypes.c_void_p(handle)
        self._closed = False

    def show(self, region_id: str, x: float, y: float) -> None:
        """Render one value into the persistent layer surface."""
        if self._closed:
            return
        error_buffer = ctypes.create_string_buffer(HUD_ERROR_SIZE)
        result = self._library.gazeebo_hud_update(
            self._handle,
            region_id.encode(),
            x,
            y,
            error_buffer,
            HUD_ERROR_SIZE,
        )
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"debug HUD update failed: {detail}"
            raise PortalError(msg)

    def close(self) -> None:
        """Destroy native Wayland resources idempotently."""
        if self._closed:
            return
        self._closed = True
        self._library.gazeebo_hud_destroy(self._handle)


class LayerShellDebugHud:
    """Update one transparent on-screen HUD at most once per second."""

    def __init__(
        self,
        surface: _Surface,
        *,
        clock: Callable[[], float] = time.monotonic,
        authorized_regions: tuple[str, ...] = (),
    ) -> None:
        """Bind the deterministic limiter to one graphical surface."""
        self._surface = surface
        self._clock = clock
        self._authorized_regions = authorized_regions
        self._last_update: float | None = None
        self._routing = "unselected"
        self._topology_quality = "unknown"
        self._model_confidence = "unknown"
        self._noise_confidence = "default"
        self._noise_alpha = 0.0
        self._noise_dead_zone = 0.0
        self._evidence = "head+face"
        self._training_region = "inactive"
        self._closed = False

    @classmethod
    def create(cls, regions: Sequence[DisplayRegion]) -> LayerShellDebugHud:
        """Create the graphical surface only for an explicit HUD request."""
        return cls(
            _NativeLayerSurface(),
            authorized_regions=tuple(region.region_id for region in regions),
        )

    def set_model_context(
        self,
        routing: str,
        topology_quality: str,
        model_confidence: str,
    ) -> None:
        """Retain safe model labels for the next rate-limited redraw."""
        self._routing = routing
        self._topology_quality = topology_quality
        self._model_confidence = model_confidence

    def set_noise_context(self, confidence: str, alpha: float, dead_zone: float) -> None:
        """Retain bounded noise-routing diagnostics for the next redraw."""
        self._noise_confidence = confidence
        self._noise_alpha = alpha
        self._noise_dead_zone = dead_zone

    def set_evidence_context(self, evidence_class: str) -> None:
        """Retain the current head-only or pupil-refined evidence class."""
        self._evidence = evidence_class

    def set_training_region(self, context: TrainingRegionStatus) -> None:
        """Retain bounded target-selection evidence for the next redraw."""
        self._training_region = (
            f"{context.region} {context.mode} CVaR90 {context.cvar90:.2f} "
            f"{context.surprise_lower:.2f}..{context.surprise_upper:.2f} "
            f"({context.observed_regions}/{context.total_regions})"
        )

    async def update(self, region_id: str, x: float, y: float) -> None:
        """Replace the HUD content at most once per one-second interval."""
        if self._closed:
            return
        now = self._clock()
        if self._last_update is not None and now - self._last_update < HUD_INTERVAL_SECONDS:
            return
        displayed_region = region_id
        if self._authorized_regions:
            authorized = ", ".join(self._authorized_regions)
            displayed_region = (
                f"{region_id}; authorized: {authorized}; "
                f"model: {self._routing}; confidence: {self._model_confidence}; "
                f"topology: {self._topology_quality}; evidence: {self._evidence}; "
                f"training: {self._training_region}; noise: {self._noise_confidence}; "
                f"alpha: {self._noise_alpha:.2f}; dead-zone: {self._noise_dead_zone:.1f}px"
            )
        self._surface.show(displayed_region, x, y)
        self._last_update = now

    async def close(self) -> None:
        """Destroy the surface exactly once."""
        if self._closed:
            return
        self._closed = True
        self._surface.close()
