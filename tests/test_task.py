from pipegoose.task import Task


def test_task():
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    task = Task(compute=counter)

    output = task.compute()

    assert output == 1
    assert count == 1
