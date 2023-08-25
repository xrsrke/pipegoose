import torch
import torch.distributed.rpc as rpc


def remote_add(x, y):
    return x + y


def run_server():
    rpc.init_rpc(
        "worker0",
        rank=0,
        world_size=2,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://127.0.0.1:29501",
        ),
    )
    print("Server ready")
    rpc.shutdown()


def run_client():
    rpc.init_rpc(
        "worker1",
        rank=1,
        world_size=2,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://127.0.0.1:29501",
        ),
    )
    print("Client ready")
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])

    # Use RPC to execute the remote_add function on the server
    result = rpc.rpc_sync("worker0", remote_add, args=(x, y))
    print(f"Client received result: {result}")
    rpc.shutdown()


if __name__ == "__main__":
    import multiprocessing

    # Spawn two processes: one for the server and one for the client
    p1 = multiprocessing.Process(target=run_server)
    p2 = multiprocessing.Process(target=run_client)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
