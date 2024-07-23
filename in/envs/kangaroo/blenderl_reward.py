import numpy as np


def reward_function(self) -> float:

    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    # Get current platform
    # platform = np.ceil((player.xy[1] - player.h - 16) / 48)  # 0: topmost, 3: lowest platform

    # # Encourage moving to the child
    # if platform % 2 == 0:  # even platform, encourage left movement
    #     reward = - player.dx
    # else:  # encourage right movement
    #     reward = player.dx
    # # Encourage upward movement
    # reward -= player.dy / 5
    # print(reward)
    # dx will be -110 when moving back to the starting point (left bottom) from the child (right top)
    # BUG: if middle. no reward
    # print("dx: ", player.dx, "dy: ", player.dy)
    if player.dy > 30:
    # if abs(reward) > 100: # level end
        reward = 99
    else:
        reward = 0
    return reward