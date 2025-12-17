"""Unit tests for Location and SimulationBoundary classes."""

import pytest

from aircompsim.entities.location import Location, SimulationBoundary


class TestLocation:
    """Tests for the Location class."""

    def test_location_creation(self):
        """Test basic location creation."""
        loc = Location(x=100, y=200, z=50)

        assert loc.x == 100
        assert loc.y == 200
        assert loc.z == 50

    def test_location_default_z(self):
        """Test default z-coordinate."""
        loc = Location(x=100, y=200)

        assert loc.z == 0.0

    def test_location_str(self):
        """Test string representation."""
        loc = Location(x=100.5, y=200.3, z=0)

        assert "(100.50, 200.30, 0.00)" in str(loc)

    def test_location_equality(self):
        """Test location equality comparison."""
        loc1 = Location(x=100, y=200, z=50)
        loc2 = Location(x=100, y=200, z=50)
        loc3 = Location(x=100, y=200, z=60)

        assert loc1 == loc2
        assert loc1 != loc3

    def test_location_hash(self):
        """Test location is hashable."""
        loc1 = Location(x=100, y=200, z=50)
        loc2 = Location(x=100, y=200, z=50)

        # Same locations should have same hash
        assert hash(loc1) == hash(loc2)

        # Should be usable in sets
        locations = {loc1, loc2}
        assert len(locations) == 1

    def test_terrestrial_property(self):
        """Test terrestrial (2D) coordinates."""
        loc = Location(x=100, y=200, z=50)

        assert loc.terrestrial == (100, 200)

    def test_coordinates_property(self):
        """Test full 3D coordinates."""
        loc = Location(x=100, y=200, z=50)

        assert loc.coordinates == (100, 200, 50)

    def test_euclidean_distance_2d(self):
        """Test 2D Euclidean distance calculation."""
        loc1 = Location(x=0, y=0, z=0)
        loc2 = Location(x=3, y=4, z=100)  # z ignored

        distance = Location.euclidean_distance_2d(loc1, loc2)

        assert distance == pytest.approx(5.0)

    def test_euclidean_distance_3d(self):
        """Test 3D Euclidean distance calculation."""
        loc1 = Location(x=0, y=0, z=0)
        loc2 = Location(x=1, y=2, z=2)

        distance = Location.euclidean_distance_3d(loc1, loc2)

        assert distance == pytest.approx(3.0)

    def test_distance_to_2d(self):
        """Test distance_to method with 2D."""
        loc1 = Location(x=0, y=0, z=0)
        loc2 = Location(x=6, y=8, z=0)

        distance = loc1.distance_to(loc2, use_3d=False)

        assert distance == pytest.approx(10.0)

    def test_distance_to_3d(self):
        """Test distance_to method with 3D."""
        loc1 = Location(x=0, y=0, z=0)
        loc2 = Location(x=1, y=2, z=2)

        distance = loc1.distance_to(loc2, use_3d=True)

        assert distance == pytest.approx(3.0)

    def test_distance_to_invalid_type(self):
        """Test distance_to raises TypeError for invalid input."""
        loc = Location(x=0, y=0, z=0)

        with pytest.raises(TypeError):
            loc.distance_to("not a location")

    def test_random_within(self):
        """Test random location generation."""
        # Generate multiple to test randomness
        locations = [Location.random_within(max_x=100, max_y=100) for _ in range(10)]

        for loc in locations:
            assert 0 <= loc.x <= 100
            assert 0 <= loc.y <= 100

        # Check they're not all identical
        unique_x = {loc.x for loc in locations}
        assert len(unique_x) > 1

    def test_random_within_with_min(self):
        """Test random location with minimum bounds."""
        loc = Location.random_within(max_x=100, max_y=100, min_x=50, min_y=50, altitude=200)

        assert 50 <= loc.x <= 100
        assert 50 <= loc.y <= 100
        assert loc.z == 200

    def test_random_within_invalid_bounds(self):
        """Test random_within raises error for invalid bounds."""
        with pytest.raises(ValueError):
            Location.random_within(max_x=50, max_y=100, min_x=100, min_y=0)

    def test_copy(self):
        """Test location copy."""
        loc1 = Location(x=100, y=200, z=50)
        loc2 = loc1.copy()

        assert loc1 == loc2
        assert loc1 is not loc2

    def test_move_towards(self):
        """Test move towards a target."""
        start = Location(x=0, y=0, z=0)
        target = Location(x=10, y=0, z=0)

        new_loc = start.move_towards(target, distance=5)

        assert new_loc.x == pytest.approx(5.0)
        assert new_loc.y == pytest.approx(0.0)

    def test_move_towards_overshoots(self):
        """Test move towards doesn't overshoot target."""
        start = Location(x=0, y=0, z=0)
        target = Location(x=5, y=0, z=0)

        # Try to move further than destination
        new_loc = start.move_towards(target, distance=100)

        assert new_loc.x == pytest.approx(5.0)
        assert new_loc.y == pytest.approx(0.0)

    def test_move_towards_at_target(self):
        """Test move towards when already at target."""
        loc = Location(x=5, y=5, z=0)

        new_loc = loc.move_towards(loc, distance=10)

        assert new_loc == loc


class TestSimulationBoundary:
    """Tests for SimulationBoundary class."""

    def test_boundary_creation(self):
        """Test basic boundary creation."""
        boundary = SimulationBoundary(max_x=400, max_y=400, max_z=200)

        assert boundary.max_x == 400
        assert boundary.max_y == 400
        assert boundary.max_z == 200
        assert boundary.min_x == 0
        assert boundary.min_y == 0
        assert boundary.min_z == 0

    def test_contains_inside(self):
        """Test location inside boundary."""
        boundary = SimulationBoundary(max_x=100, max_y=100, max_z=100)
        loc = Location(x=50, y=50, z=50)

        assert boundary.contains(loc)

    def test_contains_on_border(self):
        """Test location on boundary edge."""
        boundary = SimulationBoundary(max_x=100, max_y=100, max_z=100)
        loc = Location(x=100, y=100, z=100)

        assert boundary.contains(loc)

    def test_contains_outside(self):
        """Test location outside boundary."""
        boundary = SimulationBoundary(max_x=100, max_y=100, max_z=100)
        loc = Location(x=150, y=50, z=50)

        assert not boundary.contains(loc)

    def test_clamp_inside(self):
        """Test clamping location already inside."""
        boundary = SimulationBoundary(max_x=100, max_y=100, max_z=100)
        loc = Location(x=50, y=50, z=50)

        clamped = boundary.clamp(loc)

        assert clamped == loc

    def test_clamp_outside(self):
        """Test clamping location outside boundary."""
        boundary = SimulationBoundary(max_x=100, max_y=100, max_z=100)
        loc = Location(x=150, y=-50, z=200)

        clamped = boundary.clamp(loc)

        assert clamped.x == 100
        assert clamped.y == 0
        assert clamped.z == 100

    def test_random_location(self):
        """Test random location generation."""
        boundary = SimulationBoundary(max_x=100, max_y=100, max_z=100)

        for _ in range(10):
            loc = boundary.random_location()
            assert boundary.contains(loc)
