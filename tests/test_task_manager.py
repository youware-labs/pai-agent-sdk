"""Tests for Task and TaskManager."""

from pai_agent_sdk.context import Task, TaskManager, TaskStatus


class TestTask:
    """Tests for Task model."""

    def test_task_creation(self) -> None:
        """Test basic task creation."""
        task = Task(
            id="1",
            subject="Test task",
            description="Test description",
        )
        assert task.id == "1"
        assert task.subject == "Test task"
        assert task.description == "Test description"
        assert task.status == TaskStatus.PENDING
        assert task.active_form is None
        assert task.owner is None
        assert task.blocks == []
        assert task.blocked_by == []
        assert task.metadata == {}

    def test_task_is_blocked(self) -> None:
        """Test is_blocked method."""
        task = Task(id="1", subject="Test", description="Test")
        assert task.is_blocked() is False

        task.blocked_by = ["2"]
        assert task.is_blocked() is True


class TestTaskManager:
    """Tests for TaskManager."""

    def test_create_task(self) -> None:
        """Test task creation."""
        manager = TaskManager()
        task = manager.create(
            subject="Implement feature",
            description="Add new functionality",
            active_form="Implementing feature",
        )
        assert task.id == "1"
        assert task.subject == "Implement feature"
        assert task.description == "Add new functionality"
        assert task.active_form == "Implementing feature"
        assert task.status == TaskStatus.PENDING

    def test_create_multiple_tasks(self) -> None:
        """Test sequential task ID generation."""
        manager = TaskManager()
        task1 = manager.create("Task 1", "Description 1")
        task2 = manager.create("Task 2", "Description 2")
        task3 = manager.create("Task 3", "Description 3")

        assert task1.id == "1"
        assert task2.id == "2"
        assert task3.id == "3"

    def test_get_task(self) -> None:
        """Test task retrieval."""
        manager = TaskManager()
        created = manager.create("Test", "Description")

        task = manager.get(created.id)
        assert task is not None
        assert task.id == created.id
        assert task.subject == "Test"

        # Non-existent task
        assert manager.get("999") is None

    def test_update_task_status(self) -> None:
        """Test updating task status."""
        manager = TaskManager()
        task = manager.create("Test", "Description")

        updated = manager.update(task.id, status=TaskStatus.IN_PROGRESS)
        assert updated is not None
        assert updated.status == TaskStatus.IN_PROGRESS

        updated = manager.update(task.id, status=TaskStatus.COMPLETED)
        assert updated is not None
        assert updated.status == TaskStatus.COMPLETED

    def test_update_task_fields(self) -> None:
        """Test updating various task fields."""
        manager = TaskManager()
        task = manager.create("Original", "Original description")

        updated = manager.update(
            task.id,
            subject="Updated",
            description="Updated description",
            active_form="Updating",
            owner="Alice",
            metadata={"priority": "high"},
        )

        assert updated is not None
        assert updated.subject == "Updated"
        assert updated.description == "Updated description"
        assert updated.active_form == "Updating"
        assert updated.owner == "Alice"
        assert updated.metadata == {"priority": "high"}

    def test_update_nonexistent_task(self) -> None:
        """Test updating non-existent task returns None."""
        manager = TaskManager()
        result = manager.update("999", status=TaskStatus.COMPLETED)
        assert result is None

    def test_add_blocking_relationship(self) -> None:
        """Test adding blocking relationships."""
        manager = TaskManager()
        task1 = manager.create("Task 1", "Description")
        task2 = manager.create("Task 2", "Description")

        # Task 1 blocks Task 2
        manager.update(task1.id, add_blocks=[task2.id])

        # Verify bidirectional relationship
        assert task2.id in task1.blocks
        assert task1.id in task2.blocked_by
        assert task2.is_blocked()

    def test_add_blocked_by_relationship(self) -> None:
        """Test adding blocked_by relationships."""
        manager = TaskManager()
        task1 = manager.create("Task 1", "Description")
        task2 = manager.create("Task 2", "Description")

        # Task 2 is blocked by Task 1
        manager.update(task2.id, add_blocked_by=[task1.id])

        # Verify bidirectional relationship
        assert task1.id in task2.blocked_by
        assert task2.id in task1.blocks
        assert task2.is_blocked()

    def test_completion_resolves_dependencies(self) -> None:
        """Test that completing a task unblocks dependent tasks."""
        manager = TaskManager()
        task1 = manager.create("Task 1", "Description")
        task2 = manager.create("Task 2", "Description")
        task3 = manager.create("Task 3", "Description")

        # Task 2 and 3 blocked by Task 1
        manager.update(task2.id, add_blocked_by=[task1.id])
        manager.update(task3.id, add_blocked_by=[task1.id])

        assert task2.is_blocked()
        assert task3.is_blocked()

        # Complete Task 1
        manager.update(task1.id, status=TaskStatus.COMPLETED)

        # Task 2 and 3 should be unblocked
        assert not task2.is_blocked()
        assert not task3.is_blocked()
        assert task1.id not in task2.blocked_by
        assert task1.id not in task3.blocked_by

    def test_list_all_tasks(self) -> None:
        """Test listing all tasks."""
        manager = TaskManager()
        manager.create("Task 1", "Description")
        manager.create("Task 2", "Description")
        manager.create("Task 3", "Description")

        tasks = manager.list_all()
        assert len(tasks) == 3
        # Should be sorted by ID
        assert [t.id for t in tasks] == ["1", "2", "3"]

    def test_list_all_empty(self) -> None:
        """Test listing tasks when none exist."""
        manager = TaskManager()
        tasks = manager.list_all()
        assert tasks == []

    def test_export_and_restore(self) -> None:
        """Test exporting and restoring task state."""
        manager = TaskManager()
        task1 = manager.create("Task 1", "Description 1", active_form="Working on 1")
        task2 = manager.create("Task 2", "Description 2")
        manager.update(task2.id, add_blocked_by=[task1.id])
        manager.update(task1.id, status=TaskStatus.IN_PROGRESS)

        # Export
        exported = manager.export_tasks()
        assert "1" in exported
        assert "2" in exported

        # Restore to new manager
        restored_manager = TaskManager.from_exported(exported)

        # Verify restored state
        assert len(restored_manager.tasks) == 2
        restored_task1 = restored_manager.get("1")
        restored_task2 = restored_manager.get("2")

        assert restored_task1 is not None
        assert restored_task1.subject == "Task 1"
        assert restored_task1.status == TaskStatus.IN_PROGRESS
        assert restored_task1.active_form == "Working on 1"

        assert restored_task2 is not None
        assert restored_task2.subject == "Task 2"
        assert task1.id in restored_task2.blocked_by

    def test_restored_manager_continues_id_sequence(self) -> None:
        """Test that restored manager continues ID sequence."""
        manager = TaskManager()
        manager.create("Task 1", "Description")
        manager.create("Task 2", "Description")

        exported = manager.export_tasks()
        restored = TaskManager.from_exported(exported)

        # New task should get ID 3
        new_task = restored.create("Task 3", "Description")
        assert new_task.id == "3"

    def test_complex_dependency_chain(self) -> None:
        """Test complex dependency resolution."""
        manager = TaskManager()
        task1 = manager.create("Implement API", "Create REST endpoints")
        task2 = manager.create("Write tests", "Unit tests for API")
        task3 = manager.create("Update docs", "API documentation")

        # Chain: task1 -> task2 -> task3
        manager.update(task2.id, add_blocked_by=[task1.id])
        manager.update(task3.id, add_blocked_by=[task1.id, task2.id])

        assert not task1.is_blocked()
        assert task2.is_blocked()
        assert task3.is_blocked()

        # Complete task1
        manager.update(task1.id, status=TaskStatus.COMPLETED)
        assert not task2.is_blocked()
        assert task3.is_blocked()  # Still blocked by task2

        # Complete task2
        manager.update(task2.id, status=TaskStatus.COMPLETED)
        assert not task3.is_blocked()  # Now unblocked
