from __future__ import annotations

import cv2

from .types import ArucoDictType, MarkerCard


class ArucoMarkerGenerator:
    """Creates printable marker images, optionally with a white quiet-zone border."""

    def __init__(self, dict_type: ArucoDictType = ArucoDictType.DICT_4X4_50):
        self.dict_type = dict_type
        self._dictionary = dict_type.dictionary()

    def generate(self, marker_id: int, marker_length_m: float = 0.1875,
                 marker_px: int = 300, border_ratio: float = 1 / 6) -> MarkerCard:
        """border_ratio: white border width per side as a fraction of the marker side.
        Use 0 for no border (detection then depends on the background contrast)."""
        inner = cv2.aruco.generateImageMarker(self._dictionary, marker_id, marker_px)
        b = round(marker_px * border_ratio)
        card = cv2.copyMakeBorder(inner, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
        return MarkerCard(
            image=cv2.cvtColor(card, cv2.COLOR_GRAY2BGR),
            dict_type=self.dict_type,
            marker_id=marker_id,
            marker_length_m=marker_length_m,
            card_length_m=marker_length_m * (1 + 2 * b / marker_px),
        )

    def save(self, card: MarkerCard, path: str) -> None:
        cv2.imwrite(path, card.image)
