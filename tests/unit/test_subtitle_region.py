"""Unit tests for Subtitle_Region dataclass."""

import pytest
from src.video_text_translator.models import Subtitle_Region


class TestSubtitleRegionCreation:
    """Test valid construction of Subtitle_Region."""

    def test_valid_region(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.x == 10
        assert r.y == 20
        assert r.width == 100
        assert r.height == 50

    def test_zero_origin(self):
        r = Subtitle_Region(x=0, y=0, width=1, height=1)
        assert r.x == 0
        assert r.y == 0

    def test_frozen(self):
        r = Subtitle_Region(x=0, y=0, width=10, height=10)
        with pytest.raises(AttributeError):
            r.x = 5  # type: ignore[misc]


class TestSubtitleRegionValidation:
    """Test __post_init__ validation constraints."""

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Subtitle_Region(x=-1, y=0, width=10, height=10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Subtitle_Region(x=0, y=-1, width=10, height=10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Subtitle_Region(x=0, y=0, width=0, height=10)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Subtitle_Region(x=0, y=0, width=-5, height=10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Subtitle_Region(x=0, y=0, width=10, height=0)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Subtitle_Region(x=0, y=0, width=10, height=-3)


class TestContainsPoint:
    """Test the contains_point method."""

    def test_point_inside(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.contains_point(50.0, 40.0) is True

    def test_point_outside_right(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.contains_point(200.0, 40.0) is False

    def test_point_outside_below(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.contains_point(50.0, 200.0) is False

    def test_point_outside_left(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.contains_point(5.0, 40.0) is False

    def test_point_outside_above(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.contains_point(50.0, 10.0) is False

    def test_point_on_left_edge(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.contains_point(10.0, 40.0) is True

    def test_point_on_right_edge(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.contains_point(110.0, 40.0) is True

    def test_point_on_top_edge(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.contains_point(50.0, 20.0) is True

    def test_point_on_bottom_edge(self):
        r = Subtitle_Region(x=10, y=20, width=100, height=50)
        assert r.contains_point(50.0, 70.0) is True
