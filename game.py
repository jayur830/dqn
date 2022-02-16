import os
import numpy as np

table = np.asarray([
    [None, None, None],
    [None, None, None],
    [None, None, None]
])


def print_table():
    os.system("cls")
    for i in range(3):
        for j in range(3):
            if table[i, j] == None:
                value = "."
            elif table[i, j] == True:
                value = "O"
            elif table[i, j] == False:
                value = "X"
            print(value, end="")
        print()


def done():
    if table[0, 0] is not None and table[0, 0] == table[0, 1] == table[0, 2]:
        return True, "O" if table[0, 0] else "X"
    if table[1, 0] is not None and table[1, 0] == table[1, 1] == table[1, 2]:
        return True, "O" if table[1, 0] else "X"
    if table[2, 0] is not None and table[2, 0] == table[2, 1] == table[2, 2]:
        return True, "O" if table[2, 0] else "X"
    elif table[0, 0] is not None and table[0, 0] == table[1, 0] == table[2, 0]:
        return True, "O" if table[0, 0] else "X"
    elif table[0, 1] is not None and table[0, 1] == table[1, 1] == table[2, 1]:
        return True, "O" if table[0, 1] else "X"
    elif table[0, 2] is not None and table[0, 2] == table[1, 2] == table[2, 2]:
        return True, "O" if table[0, 2] else "X"
    elif table[0, 0] is not None and table[0, 0] == table[1, 1] == table[2, 2]:
        return True, "O" if table[0, 0] else "X"
    elif table[2, 0] is not None and table[2, 0] == table[1, 1] == table[0, 2]:
        return True, "O" if table[2, 0] else "X"
    else:
        if table[0, 0] is None or table[0, 1] is None or table[0, 2] is None \
                or table[1, 0] is None or table[1, 1] is None or table[1, 2] is None \
                or table[2, 0] is None or table[2, 1] is None or table[2, 2] is None:
            return False, None
        return True, None


def game():
    player, state = False, None
    while True:
        print_table()
        num = int(input())
        if num == 1:
            table[2, 0] = player
        elif num == 2:
            table[2, 1] = player
        elif num == 3:
            table[2, 2] = player
        elif num == 4:
            table[1, 0] = player
        elif num == 5:
            table[1, 1] = player
        elif num == 6:
            table[1, 2] = player
        elif num == 7:
            table[0, 0] = player
        elif num == 8:
            table[0, 1] = player
        elif num == 9:
            table[0, 2] = player
        player = not player
        state, winner = done()
        if state:
            break
    print_table()
    print(f"Winner: {'-' if winner is None else winner}")


if __name__ == "__main__":
    game()
