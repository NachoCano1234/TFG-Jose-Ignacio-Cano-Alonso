import itertools
import numpy as np #Cambio Nacho MetaRL. Se importa numpy para usar la función tanh y así normalizar las recompensas de Kirbie a un rango entre -1 y 1.
from pyboy import WindowEvent
from AISettings.AISettingsInterface import AISettingsInterface
from AISettings.AISettingsInterface import Config


class GameState:
    def __init__(self, pyboy):
        #game_wrapper = pyboy.game_wrapper()
        self.boss_health = pyboy.get_memory_value(0xD093)
        self.screen_x_position = pyboy.get_memory_value(0xD053)
        self.kirby_x_position = pyboy.get_memory_value(0xD05C)
        self.kirby_y_position = pyboy.get_memory_value(0xD05D)
        self.game_state = pyboy.get_memory_value(0xD02C)
        scx = pyboy.botsupport_manager().screen().tilemap_position_list()[16][0]
        self.level_progress = self.screen_x_position * 16 + (scx - 7) % 16 + self.kirby_x_position
        #self.health = game_wrapper.health #Quitamos las conexiones con el wrapper debido a problemas en windows
        #self.lives_left = game_wrapper.lives_left #Quitamos las conexiones con el wrapper debido a problemas en windows
        self.health = pyboy.get_memory_value(0xD086)
        self.lives_left = pyboy.get_memory_value(0xD089)
        #self.score = game_wrapper.score #Quitamos las conexiones con el wrapper debido a problemas en windows
        self.score = (pyboy.get_memory_value(0xD070) + pyboy.get_memory_value(0xD071) * 100)


class KirbyAI(AISettingsInterface):
    '''
    def GetReward(self, previous_kirby: GameState, pyboy):
        current_kirby = GameState(pyboy)

        if current_kirby.boss_health == 0 and previous_kirby.boss_health > 0:
            return 10000

        if current_kirby.boss_health < previous_kirby.boss_health:
            return 1000

        if current_kirby.health < previous_kirby.health and current_kirby.health == 1:
            return -100

        if current_kirby.health == 0 and previous_kirby.health != 0:
            return -1000

        if current_kirby.health > 0 and current_kirby.game_state == 6 and previous_kirby.game_state != 6:  # if reached warpstar
            return 1000

        if not self.IsBossActive(pyboy) and current_kirby.game_state != 6:  # if boss is dead or not active, punish for not moving right
            if current_kirby.kirby_x_position < previous_kirby.kirby_x_position:  # moving left
                return -1

            if current_kirby.level_progress != previous_kirby.level_progress and current_kirby.kirby_x_position == 68:  # moving most left
                return -5

            if current_kirby.level_progress == previous_kirby.level_progress:  # standing still
                return -1

            if current_kirby.kirby_x_position == 76:  # moving most right
                return 5
            return 1  # moving right
        else:
            if current_kirby.score>previous_kirby.score:
                return 100
        return 0
    '''
    def GetReward(self, previous_kirby: GameState, pyboy): #Cambio por Nacho MetaRL. Como las recompensas de Kirbie eran enormes en comparación con las de Mario y ahora el modelo es multitarea, se han reducido todas las recompensas a la centésima parte de su valor original, para que el rango de recompensas sea más parecido al de Mario y que así no perjudiquen al aprendizaje del modelo meta.
        current_kirby = GameState(pyboy)

        reward = 0

        if current_kirby.boss_health == 0 and previous_kirby.boss_health > 0:
            reward = 10000

        elif current_kirby.boss_health < previous_kirby.boss_health:
            reward = 1000

        elif current_kirby.health < previous_kirby.health and current_kirby.health == 1:
            reward = -100

        elif current_kirby.health == 0 and previous_kirby.health != 0:
            reward = -1000

        elif current_kirby.health > 0 and current_kirby.game_state == 6 and previous_kirby.game_state != 6:
            reward = 1000

        elif not self.IsBossActive(pyboy) and current_kirby.game_state != 6:
            if current_kirby.kirby_x_position < previous_kirby.kirby_x_position:
                reward = -1
            elif current_kirby.level_progress != previous_kirby.level_progress and current_kirby.kirby_x_position == 68:
                reward = -5
            elif current_kirby.level_progress == previous_kirby.level_progress:
                reward = -1
            elif current_kirby.kirby_x_position == 76:
                reward = 5
            else:
                reward = 1
        else:
            if current_kirby.score > previous_kirby.score:
                reward = 100

        reward = reward / 100.0  # Normalizar la recompensa a un rango entre -1 y 1 y asi evitar outliers.

        return reward

    def GetActions(self):
        baseActions = [WindowEvent.PRESS_BUTTON_A,
                       WindowEvent.PRESS_BUTTON_B,
                       WindowEvent.PRESS_ARROW_UP,
                       WindowEvent.PRESS_ARROW_DOWN,
                       WindowEvent.PRESS_ARROW_LEFT,
                       WindowEvent.PRESS_ARROW_RIGHT
                       ]

        totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
        withoutRepeats = []

        for combination in totalActionsWithRepeats:
            reversedCombination = combination[::-1]
            if (reversedCombination not in withoutRepeats):
                withoutRepeats.append(combination)

        filteredActions = [[action] for action in baseActions] + withoutRepeats

        return filteredActions

    def PrintGameState(self, pyboy):
        pass

    def GetGameState(self, pyboy) -> GameState:
        return GameState(pyboy)

    def GetLength(self, pyboy):
        return self.GetGameState(pyboy).boss_health

    def IsBossActive(self, pyboy):
        if self.GetGameState(pyboy).boss_health > 0:
            return True
        return False

    def GetHyperParameters(self) -> Config:
        config = Config()
        config.exploration_rate_decay = 0.9999975
        config.exploration_rate_min = 0.01
        config.deque_size = 500000
        config.batch_size = 64
        config.save_every = 2e5
        config.learning_rate_decay = 0.9999985
        config.gamma = 0.8
        config.learning_rate = 0.0002
        config.burnin = 1000
        config.sync_every = 100
        return config

    def GetBossHyperParameters(self) -> Config:
        config = self.GetHyperParameters()
        config.exploration_rate_decay = 0.99999975
        return config

    def IsDone(self, pyboy): #Quitamos las conexiones con el wrapper debido a problemas en windows

        health = pyboy.get_memory_value(0xD086)

        return health == 0