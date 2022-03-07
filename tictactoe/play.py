import os
import pygame
import numpy as np
import time

from tensorflow.keras.models import load_model
from tictactoe.commons import agent, enemy as human, empty

indices = np.array([[i, j] for i in range(3) for j in range(3)])


def result(state):
    vertical_sum_state, horizontal_sum_state = state.sum(axis=0), state.sum(axis=1)
    if human * 3 in vertical_sum_state or human * 3 in horizontal_sum_state \
            or state[0, 0] + state[1, 1] + state[2, 2] == human * 3 \
            or state[0, 2] + state[1, 1] + state[2, 0] == human * 3:
        return True, human
    elif agent * 3 in vertical_sum_state or agent * 3 in horizontal_sum_state \
            or state[0, 0] + state[1, 1] + state[2, 2] == agent * 3 \
            or state[0, 2] + state[1, 1] + state[2, 0] == agent * 3:
        return True, agent
    elif np.where(state.reshape(-1) == empty)[0].shape[0] == 0:
        return True, 0, state
    else:
        return False, 0


def affine(_state):
    w, h = _state.shape[1], _state.shape[0]
    a = np.zeros(shape=(2 * h - 1, w))
    b = np.zeros(shape=(2 * h - 1, w))
    for col in range(w):
        a[col:col + h, col] = _state[:, col]
        b[-h - col:2 * h - 1 - col, col] = _state[:, col]
    return a, b


def game_init():
    screen = pygame.display.set_mode(size=(cell_size * 5, cell_size * 5))

    for i in range(5):
        pygame.draw.line(
            surface=screen,
            color=(50, 50, 50),
            start_pos=(cell_size, cell_size * i),
            end_pos=(cell_size * 4, cell_size * i))
    for i in range(5):
        pygame.draw.line(
            surface=screen,
            color=(50, 50, 50),
            start_pos=(cell_size * i, cell_size),
            end_pos=(cell_size * i, cell_size * 4))

    pygame.display.update()

    board = np.ones(shape=(3, 3)) * empty

    font = pygame.font.Font("RIX모던고딕B.TTF", 18)
    text = font.render("AI", True, RED)
    screen.blit(text, dest=(int(4 * cell_size * .15), int(cell_size * .35)))
    text = font.render("Human", True, GREEN)
    screen.blit(text, dest=(int(4 * cell_size * .9), int(cell_size * .35)))
    pygame.display.flip()

    return screen, board, human, agent


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    pygame.init()

    run = True
    agent_model = load_model("tictactoe.h5")
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    cell_size = 100

    screen, board, human, agent = game_init()
    done = False

    while run:
        for event in pygame.event.get():
            if done and event.type == pygame.KEYUP and event.key == 32:
                screen, board, human, agent = game_init()
                done = False
            elif event.type == pygame.MOUSEBUTTONUP:
                if done:
                    continue
                center = (cell_size * (event.pos[0] // cell_size + .5), cell_size * (event.pos[1] // cell_size + .5))
                if 0 not in center and cell_size * 4 not in center:
                    index = ((event.pos[0] // cell_size) - 1, (event.pos[1] // cell_size) - 1)
                    board[index[1], index[0]] = human
                    pygame.draw.circle(
                        surface=screen,
                        color=GREEN,
                        center=center,
                        radius=int(cell_size * .45),
                        width=4)
                    pygame.display.flip()
                    done, winner = result(board)
                    if done:
                        font = pygame.font.Font("RIX모던고딕B.TTF", 18)
                        if winner == agent:
                            text = font.render("LOSE", True, (255, 0, 0))
                            screen.blit(text, dest=(int(4 * cell_size * .5), int(cell_size * .35)))
                        elif winner == human:
                            text = font.render("WIN", True, (0, 0, 255))
                            screen.blit(text, dest=(int(4 * cell_size * .5), int(cell_size * .35)))
                        else:
                            text = font.render("DRAW", True, (0, 255, 0))
                            screen.blit(text, dest=(int(4 * cell_size * .5), int(cell_size * .35)))
                        pygame.display.flip()
                    else:
                        time.sleep(1 + np.random.random())
                        indexes = np.where(board.reshape(-1) != empty)[0]
                        q_output = np.array(agent_model(board.reshape((1,) + board.shape + (1,)))).reshape(-1)
                        q_output = (q_output - np.min(q_output)) / (np.max(q_output) - np.min(q_output))
                        q_output[indexes] = 0
                        action = np.argmax(q_output)
                        board[indices[action, 0], indices[action, 1]] = agent
                        pygame.draw.line(
                            surface=screen,
                            color=RED,
                            start_pos=(cell_size * (indices[action, 1] + 1), cell_size * (indices[action, 0] + 1)),
                            end_pos=(cell_size * (indices[action, 1] + 2), cell_size * (indices[action, 0] + 2)),
                            width=4)
                        pygame.draw.line(
                            surface=screen,
                            color=RED,
                            start_pos=(cell_size * (indices[action, 1] + 2), cell_size * (indices[action, 0] + 1)),
                            end_pos=(cell_size * (indices[action, 1] + 1), cell_size * (indices[action, 0] + 2)),
                            width=4)
                        pygame.display.flip()
                        done, winner = result(board)
                        if done:
                            font = pygame.font.Font("RIX모던고딕B.TTF", 18)
                            if winner == agent:
                                text = font.render("LOSE", True, (0, 0, 255))
                                screen.blit(text, dest=(int(4 * cell_size * .5), int(cell_size * .35)))
                            elif winner == human:
                                text = font.render("WIN", True, (0, 0, 255))
                                screen.blit(text, dest=(int(4 * cell_size * .5), int(cell_size * .35)))
                            else:
                                text = font.render("DRAW", True, (0, 0, 255))
                                screen.blit(text, dest=(int(4 * cell_size * .5), int(cell_size * .35)))
                            pygame.display.flip()
            elif event.type == pygame.QUIT:
                run = False

    pygame.quit()
