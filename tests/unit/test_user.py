"""Unit tests for User entity."""

import pytest

from aircompsim.entities.location import Location
from aircompsim.entities.task import Application, ApplicationType
from aircompsim.entities.user import FlyingUser, MobileUser, User

# Create a test application type
TEST_APP_TYPE = ApplicationType(
    name="TestApp",
    cpu_cycle=100.0,
    worst_delay=1.0,
    best_delay=0.1,
    interarrival_time=0.5,
)


class TestUser:
    """Tests for User class."""

    @pytest.fixture(autouse=True)
    def reset_users(self):
        """Reset user registry before each test."""
        User.reset_all()
        yield
        User.reset_all()

    def test_user_creation(self):
        """Test basic user creation."""
        loc = Location(100.0, 200.0, 0.0)
        user = User(location=loc)

        assert user.id > 0
        assert user.location == loc
        assert not user.is_moving
        assert len(user.applications) == 0

    def test_user_unique_ids(self):
        """Test that users get unique IDs."""
        user1 = User(location=Location(0, 0, 0))
        user2 = User(location=Location(10, 10, 0))
        user3 = User(location=Location(20, 20, 0))

        assert user1.id != user2.id
        assert user2.id != user3.id
        assert len(User.get_all()) == 3

    def test_user_equality_and_hash(self):
        """Test user equality and hashing."""
        user1 = User(location=Location(0, 0, 0))
        user2 = User(location=Location(0, 0, 0))

        assert user1 != user2  # Different IDs
        assert hash(user1) != hash(user2)

    def test_add_application(self):
        """Test adding application to user."""
        user = User(location=Location(0, 0, 0))
        app = Application(app_type=TEST_APP_TYPE, start_time=0.0)

        user.add_application(app)

        assert len(user.applications) == 1
        assert app in user.applications
        assert app.user_id == user.id

    def test_remove_application(self):
        """Test removing application from user."""
        user = User(location=Location(0, 0, 0))
        app = Application(app_type=TEST_APP_TYPE, start_time=0.0)
        user.add_application(app)

        result = user.remove_application(app)

        assert result is True
        assert len(user.applications) == 0

    def test_remove_nonexistent_application(self):
        """Test removing application that doesn't exist."""
        user = User(location=Location(0, 0, 0))
        app = Application(app_type=TEST_APP_TYPE, start_time=0.0)

        result = user.remove_application(app)

        assert result is False

    def test_get_location(self):
        """Test get_location method."""
        loc = Location(50.0, 75.0, 0.0)
        user = User(location=loc)

        assert user.get_location() == loc
        assert user.current_location == loc

    def test_current_location_setter(self):
        """Test setting current location."""
        user = User(location=Location(0, 0, 0))
        new_loc = Location(100, 100, 0)

        user.current_location = new_loc

        assert user.location == new_loc

    def test_compute_movement_duration(self):
        """Test movement duration calculation."""
        user = User(location=Location(0, 0, 0), speed=2.0)
        destination = Location(10, 0, 0)

        duration = user.compute_movement_duration(destination)

        assert duration == pytest.approx(5.0)  # 10m / 2m/s = 5s

    def test_get_next_location(self):
        """Test random next location generation."""
        user = User(location=Location(100, 100, 0))

        next_loc = user.get_next_location(radius=50)

        # Should be within radius
        distance = Location.euclidean_distance_2d(user.location, next_loc)
        assert distance <= 50 * 1.42  # Allow for diagonal movement

    def test_get_next_location_with_boundary(self):
        """Test next location with boundary constraints."""
        user = User(location=Location(10, 10, 0))

        next_loc = user.get_next_location(radius=50, boundary=(100, 100))

        assert 0 <= next_loc.x <= 100
        assert 0 <= next_loc.y <= 100

    def test_move_to(self):
        """Test moving user to new location."""
        user = User(location=Location(0, 0, 0))
        new_loc = Location(50, 50, 0)

        user.move_to(new_loc)

        assert user.location == new_loc
        assert len(user.trajectory) == 2  # Initial + new

    def test_start_stop_moving(self):
        """Test movement state transitions."""
        user = User(location=Location(0, 0, 0))

        assert not user.is_moving

        user.start_moving()
        assert user.is_moving

        user.stop_moving()
        assert not user.is_moving

    def test_get_qoe_no_applications(self):
        """Test QoE with no applications."""
        user = User(location=Location(0, 0, 0))

        assert user.get_qoe() == 0.0

    def test_get_statistics(self):
        """Test getting user statistics."""
        user = User(location=Location(100, 200, 0))
        user.add_application(Application(app_type=TEST_APP_TYPE, start_time=0.0))

        stats = user.get_statistics()

        assert stats["user_id"] == user.id
        assert stats["num_applications"] == 1
        assert stats["trajectory_length"] == 1
        assert "location" in stats

    def test_reset_all(self):
        """Test resetting all users."""
        User(location=Location(0, 0, 0))
        User(location=Location(10, 10, 0))

        assert len(User.get_all()) == 2

        User.reset_all()

        assert len(User.get_all()) == 0

    def test_get_all(self):
        """Test getting all users."""
        user1 = User(location=Location(0, 0, 0))
        user2 = User(location=Location(10, 10, 0))

        users = User.get_all()

        assert len(users) == 2
        assert user1 in users
        assert user2 in users

    def test_get_user_by_id(self):
        """Test getting user by ID."""
        user = User(location=Location(0, 0, 0))
        user_id = user.id

        found = User.get_user(user_id)

        assert found == user

    def test_get_user_nonexistent(self):
        """Test getting nonexistent user."""
        found = User.get_user(99999)

        assert found is None


class TestMobileUser:
    """Tests for MobileUser class."""

    @pytest.fixture(autouse=True)
    def reset_users(self):
        """Reset user registry before each test."""
        User.reset_all()
        yield
        User.reset_all()

    def test_mobile_user_creation(self):
        """Test MobileUser creation."""
        user = MobileUser(location=Location(0, 0, 0), max_speed=10.0)

        assert user.max_speed == 10.0

    def test_update_speed(self):
        """Test updating user speed."""
        user = MobileUser(location=Location(0, 0, 0), max_speed=10.0)

        user.update_speed(5.0)
        assert user.speed == 5.0

        # Should cap at max speed
        user.update_speed(15.0)
        assert user.speed == 10.0


class TestFlyingUser:
    """Tests for FlyingUser class."""

    @pytest.fixture(autouse=True)
    def reset_users(self):
        """Reset user registry before each test."""
        User.reset_all()
        yield
        User.reset_all()

    def test_flying_user_creation(self):
        """Test FlyingUser creation."""
        user = FlyingUser(location=Location(0, 0, 0), altitude=100.0)

        assert user.altitude == 100.0

    def test_flying_user_current_location(self):
        """Test FlyingUser 3D location."""
        user = FlyingUser(location=Location(10, 20, 0), altitude=50.0)

        loc = user.current_location

        assert loc.x == 10
        assert loc.y == 20
        assert loc.z == 50.0

    def test_flying_user_set_location(self):
        """Test setting FlyingUser location."""
        user = FlyingUser(location=Location(0, 0, 0), altitude=0.0)

        user.current_location = Location(100, 200, 300)

        assert user.location.x == 100
        assert user.location.y == 200
        assert user.altitude == 300
