import numpy as np
import torch
import torchgeometry as tgm
from scipy.spatial.transform import Rotation as R


def get_jacobian(net, x, noutputs):
    x = x.squeeze()
    n = x.size()[0]
    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.eye(noutputs))
    return x.grad.data


def jacobian(outputs, inputs, create_graph=False):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True,
                                    create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())


def transform(inputs, vector):
    # theta = torch.sqrt(inputs[0] ** 2 + inputs[1] ** 2 + inputs[3] ** 2)
    # u_x = inputs[0] / theta
    # u_y = inputs[1] / theta
    # u_z = inputs[2] / theta

    inputs = inputs.repeat(vector.shape[0], 1)

    rot = tgm.angle_axis_to_rotation_matrix(inputs[:, :3])[:, :3, :3].float()

    offset = inputs[:, 3:].float()

    transformed = torch.bmm(rot, vector.unsqueeze(-1)).squeeze(-1) + offset
    return transformed


def main():
    point = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.4, 0.0],
                      [0.4, 0.0, 0.0],
                      [0.4, 0.4, 0.0],
                      [0.0, 0.0, 0.4],
                      [0.0, 0.4, 0.4],
                      [0.4, 0.0, 0.4],
                      [0.4, 0.4, 0.4]
                      ])

    # rot = np.array([[0.6418311, 0, 0],
    #                 [0, 1, 0],
    #                 [0, 0, 1]])
    rot = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]).as_dcm()

    offset = np.array([[4, 5, 6]])

    transformed = np.matmul(rot, point.T).T + offset

    point = torch.from_numpy(point).float().requires_grad_(True)
    transformed = torch.from_numpy(transformed).float()

    inputs = torch.Tensor([0, 0, 1, 1, 1, 1]).requires_grad_(True)
    # inputs = inputs.repeat(8, 1)

    loss = torch.nn.L1Loss(reduction='none')

    value = 50
    count = 0
    while count < 50:
        estimate = transform(inputs=inputs, vector=point)
        residual = (estimate - transformed).view(-1)

        jac = jacobian(outputs=residual, inputs=inputs)
        # jac = jac.view(jac.shape[0] * jac.shape[1], jac.shape[2])
        if torch.abs(torch.mean(residual)).item() > value:
            print("Wrong!! ", torch.abs(torch.mean(residual)).item(), value, count)
            break
        jac = jac.squeeze()

        delta = torch.mv(torch.inverse(torch.mm(jac.T, jac) + 1e-3 * torch.diag(torch.diag(torch.mm(jac.T, jac)))),
                         torch.mv(jac.T, (estimate - transformed).view(jac.shape[0])))
        inputs = inputs - delta
        # print(inputs)

        value = torch.abs(torch.mean(residual)).item()
        count += 1
    print("No. of iterations: {}".format(count))
    print("Applied transformation:\n", transform(inputs=inputs, vector=point))
    print("True transformation:\n", transformed)
    print("Parameters:\n", inputs)

    print(torch.sum(torch.abs(transform(inputs=inputs, vector=point) - transformed)))


if __name__ == "__main__":
    main()
