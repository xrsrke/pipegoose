from pipegoose.task import Task


def test_task():
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    task = Task(func=counter)

    output = task()

    assert output == 1
    assert count == 1
