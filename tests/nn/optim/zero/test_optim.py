# from pipegoose.optim import ZeroRedundancyOptimizer


# def test_optim():
#     WORLD_SIZE = 10

#     params = []
#     sizes = [2, 3, 4] * WORLD_SIZE

#     for size in sizes:
#         params.append([torch.randn((size, 1), requires_grad=True)])

#     optim = SGD(params, lr=0.1)
#     dist_optim = ZeroRedundancyOptimizer
