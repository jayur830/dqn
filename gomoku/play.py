import os
import pygame
import numpy as np
import time

from tensorflow.keras.models import load_model
from gomoku.commons import gomoku_size, black, white, empty

indices = np.array([[i, j] for i in range(gomoku_size) for j in range(gomoku_size)])


def result(state):
    for i in range(np.shape(state)[0] - 4):
        summary = np.sum(state[i:i + 5], axis=0)
        if np.sum(summary == black * 5) > 0:
            return True, black
        elif np.sum(summary == white * 5) > 0:
            return True, white
    for j in range(state.shape[1] - 4):
        summary = np.sum(state[:, j:j + 5], axis=1)
        if np.sum(summary == black * 5) > 0:
            return True, black
        elif np.sum(summary == white * 5) > 0:
            return True, white
    state_affine1, state_affine2 = affine(state)
    for i in range(state_affine1.shape[0] - 4):
        summary = np.sum(state_affine1[i:i + 5], axis=0)
        if np.sum(summary == black * 5) > 0:
            return True, black
        elif np.sum(summary == white * 5) > 0:
            return True, white
    for j in range(state_affine1.shape[1] - 4):
        summary = np.sum(state_affine1[:, j:j + 5], axis=1)
        if np.sum(summary == black * 5) > 0:
            return True, black
        elif np.sum(summary == white * 5) > 0:
            return True, white
    for i in range(state_affine2.shape[0] - 4):
        summary = np.sum(state_affine2[i:i + 5], axis=0)
        if np.sum(summary == black * 5) > 0:
            return True, black
        elif np.sum(summary == white * 5) > 0:
            return True, white
    for j in range(state_affine2.shape[1] - 4):
        summary = np.sum(state_affine2[:, j:j + 5], axis=1)
        if np.sum(summary == black * 5) > 0:
            return True, black
        elif np.sum(summary == white * 5) > 0:
            return True, white
    return not np.any(state == empty), 0


def affine(_state):
    w, h = _state.shape[1], _state.shape[0]
    a = np.zeros(shape=(2 * h - 1, w))
    b = np.zeros(shape=(2 * h - 1, w))
    for col in range(w):
        a[col:col + h, col] = _state[:, col]
        b[-h - col:2 * h - 1 - col, col] = _state[:, col]
    return a, b


def game_init():
    screen = pygame.display.set_mode(size=(cell_size * (gomoku_size + 1), cell_size * (gomoku_size + 1)))
    screen.fill((229, 166, 68))

    for i in range(1, gomoku_size + 1):
        pygame.draw.line(
            surface=screen,
            color=(50, 50, 50),
            start_pos=(cell_size, cell_size * i),
            end_pos=(cell_size * gomoku_size, cell_size * i))
    for i in range(1, gomoku_size + 1):
        pygame.draw.line(
            surface=screen,
            color=(50, 50, 50),
            start_pos=(cell_size * i, cell_size),
            end_pos=(cell_size * i, cell_size * gomoku_size))

    pygame.display.update()

    human = [black, white][np.random.randint(2)]
    agent = white if human == black else black
    board = np.ones(shape=(gomoku_size, gomoku_size)) * empty
    first_is = [human, agent][np.random.randint(2)]
    board[gomoku_size // 2, gomoku_size // 2] = first_is
    pygame.draw.circle(
        surface=screen,
        color=BLACK if first_is == black else WHITE,
        center=(cell_size * round(gomoku_size / 2), cell_size * round(gomoku_size / 2)),
        radius=int(cell_size * .45))
    pygame.display.flip()

    if first_is == human:
        indexes = np.where(board.reshape(-1) != empty)[0]
        q_output = np.array(agent_model(board.reshape((1,) + board.shape + (1,)))).reshape(-1)
        q_output = (q_output - np.min(q_output)) / (np.max(q_output) - np.min(q_output))
        q_output[indexes] = 0
        action = np.argmax(q_output)
        board[indices[action, 0], indices[action, 1]] = agent
        pygame.draw.circle(
            surface=screen,
            color=BLACK if agent == black else WHITE,
            center=(cell_size * (indices[action, 1] + 1), cell_size * (indices[action, 0] + 1)),
            radius=int(cell_size * .45))
        pygame.display.flip()

    font = pygame.font.Font("RIX모던고딕B.TTF", 14)
    text = font.render("AI", True, BLACK if agent == black else WHITE)
    screen.blit(text, dest=(int(gomoku_size * cell_size * .15), int(cell_size * .35)))
    text = font.render("Human", True, BLACK if human == black else WHITE)
    screen.blit(text, dest=(int(gomoku_size * cell_size * .9), int(cell_size * .35)))
    if first_is == agent:
        text = font.render("First", True, BLACK if agent == black else WHITE)
        screen.blit(text, dest=(int(gomoku_size * cell_size * .25), int(cell_size * .35)))
    else:
        text = font.render("First", True, BLACK if human == black else WHITE)
        screen.blit(text, dest=(int(gomoku_size * cell_size * .75), int(cell_size * .35)))
    pygame.display.flip()

    return screen, board, human, agent


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    pygame.init()

    run = True
    agent_model = load_model("gomoku.h5")
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    cell_size = 40

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
                center = (cell_size * round(event.pos[0] / cell_size), cell_size * round(event.pos[1] / cell_size))
                if 0 not in center:
                    index = (round(event.pos[0] / cell_size) - 1, round(event.pos[1] / cell_size) - 1)
                    board[index[1], index[0]] = human
                    pygame.draw.circle(
                        surface=screen,
                        color=BLACK if human == black else WHITE,
                        center=center,
                        radius=int(cell_size * .45))
                    pygame.display.flip()
                    done, winner = result(board)
                    if done:
                        font = pygame.font.Font("RIX모던고딕B.TTF", 14)
                        if winner == agent:
                            text = font.render("LOSE", True, (255, 0, 0))
                            screen.blit(text, dest=(int(gomoku_size * cell_size * .5), int(cell_size * .35)))
                        elif winner == human:
                            text = font.render("WIN", True, (0, 0, 255))
                            screen.blit(text, dest=(int(gomoku_size * cell_size * .5), int(cell_size * .35)))
                        else:
                            text = font.render("DRAW", True, (0, 255, 0))
                            screen.blit(text, dest=(int(gomoku_size * cell_size * .5), int(cell_size * .35)))
                        pygame.display.flip()
                    else:
                        time.sleep(1 + np.random.random())
                        indexes = np.where(board.reshape(-1) != empty)[0]
                        q_output = np.array(agent_model(board.reshape((1,) + board.shape + (1,)))).reshape(-1)
                        q_output = (q_output - np.min(q_output)) / (np.max(q_output) - np.min(q_output))
                        q_output[indexes] = 0
                        action = np.argmax(q_output)
                        board[indices[action, 0], indices[action, 1]] = agent
                        pygame.draw.circle(
                            surface=screen,
                            color=BLACK if agent == black else WHITE,
                            center=(cell_size * (indices[action, 1] + 1), cell_size * (indices[action, 0] + 1)),
                            radius=int(cell_size * .45))
                        pygame.display.flip()
                        done, winner = result(board)
                        if done:
                            font = pygame.font.Font("RIX모던고딕B.TTF", 14)
                            if winner == agent:
                                text = font.render("LOSE", True, (255, 0, 0))
                                screen.blit(text, dest=(int(gomoku_size * cell_size * .5), int(cell_size * .35)))
                            elif winner == human:
                                text = font.render("WIN", True, (0, 0, 255))
                                screen.blit(text, dest=(int(gomoku_size * cell_size * .5), int(cell_size * .35)))
                            else:
                                text = font.render("DRAW", True, (0, 255, 0))
                                screen.blit(text, dest=(int(gomoku_size * cell_size * .5), int(cell_size * .35)))
                            pygame.display.flip()
            elif event.type == pygame.QUIT:
                run = False

    pygame.quit()
