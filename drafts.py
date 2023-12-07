from socket import AI_DEFAULT


def get_element(nested_list, indices):
    element = nested_list
    for index in indices:
        element = element[index]
    return element

def set_element(nested_list, indices, to):
    element = nested_list
    for i, index in enumerate(indices):
        if i == len(indices)-1:
            element[index] = to
            return

        element = element[index]

my_list = [[1, 2], [3, 4]]
indices = [0, 1]
element = get_element(my_list, indices)


def add(a, ashape, b, bshape, out, outshape):
    stack = []

    def getIdx(shape):
        return [min(dim-1, idx) for dim, idx in zip(shape, stack)]

    def nested_loop(depth: int):
        if depth == 3:
            res = get_element(a, getIdx(ashape))\
                + get_element(b, getIdx(bshape))
            set_element(
                out,
                stack,
                res
            )
            return

        for i in range(outshape[depth]):
            stack.append(i)
            nested_loop(depth+1)
            stack.pop()

    nested_loop(0)

a = [[[1],
      [2],
      [3],
      [1]],
     [[5],
      [3],
      [9],
      [2]]]
ashape = [2, 4, 1]

b = [[[1, 2, 5],
      [1, 2, 2],
      [2, 2, 5],
      [7, 2, 3]]]
bshape = [1, 4, 3]

empty = [[[None, None, None],
          [None, None, None],
          [None, None, None],
          [None, None, None]],
         [[None, None, None],
          [None, None, None],
          [None, None, None],
          [None, None, None]]]
shape = [2, 4, 3]

add(a, ashape, b, bshape, empty, shape)

print(empty)






















# a = [
#     [1, 2, 2],
# ]
# a_shape = [1, 3]

# b = [
#     [1, 1, 1],
#     [1, 1, 1],
#     [1, 1, 1],
#     [1, 1, 1],
# ]
# b_shape = [4, 3]

# bc_dir = ['right', None]

# out = []
# out_shape = []
# for l_d, r_d in zip(a_shape, b_shape):
#     out_shape.append(max(l_d, r_d))

# print(out_shape)

# for dir in bc_dir:


