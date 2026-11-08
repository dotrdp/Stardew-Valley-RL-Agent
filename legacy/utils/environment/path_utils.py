
def print_path(spatial_state, path):
    copy = spatial_state.copy()
    for point in path:
        x, y = point
        type = copy[x][y].type
        if type == "player":
            continue
        if "tool" in copy[x][y].properties:
            if "collision" in copy[x][y].properties:
                copy[x][y].type = "debug_red"
                continue
        copy[x][y].type = "debugmarker"
    res = ""
    copy = list(zip(*copy))  # Transpose the matrix for easier printing
    for x in range(len(copy)):
        for y in range(len(copy[x])):
            tile = copy[x][y]
            res += str(tile)
        res += "\n"
    print(res)
