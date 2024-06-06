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


def add(l, ashape, r, bshape, out, outshape):
    stack = []

    def getIdx(shape):
        return [min(dim-1, idx) for dim, idx in zip(shape, stack)]

    def nested_loop(depth: int):
        if depth == 3:
            res = get_element(l, getIdx(ashape))\
                + get_element(r, getIdx(bshape))
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

def add2(l, r, out, dirs):
    stack = []

    def getIndices() -> tuple[list[int], list[int]]:
        l = []
        r = []
        for broadcastdir, idx in zip(dirs, stack):
            match broadcastdir:
                case 'ltr':
                    l.append(0)
                    r.append(idx)
                case 'rtl':
                    l.append(idx)
                    r.append(0)
                case None:
                    l.append(idx)
                    r.append(idx)
                case _:
                    raise Rc<Tensor>ueError(f'huh?, got {broadcastdir=}')
        return l, r

    def nested_loop(depth: int):
        if depth == 3:
            l_idx, r_idx = getIndices()
            res = None
            try:
                res = get_element(l, l_idx) + get_element(r, r_idx)
            except:
                breakpoint()

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


l = [[[1],
      [2],
      [3],
      [1]],
     [[5],
      [3],
      [9],
      [2]]]
lshape = [2, 4, 1]

r = [[[1, 2, 5],
      [1, 2, 2],
      [2, 2, 5],
      [7, 2, 3]]]
rshape = [1, 4, 3]

empty = [[[None, None, None],
          [None, None, None],
          [None, None, None],
          [None, None, None]],
         [[None, None, None],
          [None, None, None],
          [None, None, None],
          [None, None, None]]]
outshape = [2, 4, 3]
out = copy.deepcopy(empty)
out2 = copy.deepcopy(empty)

add(l, lshape, r, rshape, out, outshape)
print(out)

broadcast_dirs = ['rtl', None, 'ltr']
add2(l, r, out2, broadcast_dirs)
print(out2)
























# l = [
#     [1, 2, 2],
# ]
# a_shape = [1, 3]

# r = [
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


