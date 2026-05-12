from asyncio import sleep
import itertools
from pyboy import WindowEvent
from AISettings.AISettingsInterface import AISettingsInterface, Config

class GameState():
    def __init__(self, pyboy):
        #game_wrapper = pyboy.game_wrapper() #Quitamos las conexiones con el wrapper debido a problemas en windows
        "Find the real level progress x"
        level_block = pyboy.get_memory_value(0xC0AB)
        # C202 Mario's X position relative to the screen
        mario_x = pyboy.get_memory_value(0xC202)
        scx = pyboy.botsupport_manager().screen(
        ).tilemap_position_list()[16][0]
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16

        self.real_x_pos = level_block * 16 + real + mario_x
        #self.time_left = game_wrapper.time_left #Quitamos las conexiones con el wrapper debido a problemas en windows
        self.time_left = pyboy.get_memory_value(0xDA01)
        #self.lives_left = game_wrapper.lives_left #Quitamos las conexiones con el wrapper debido a problemas en windows
        self.lives_left = self.lives_left = pyboy.get_memory_value(0xDA15)
        #self.score = game_wrapper.score #Quitamos las conexiones con el wrapper debido a problemas en windows #Quitamos las conexiones con el wrapper debido a problemas en windows
        self.score = (pyboy.get_memory_value(0xC0A0) + pyboy.get_memory_value(0xC0A1) * 10)
        #self._level_progress_max = max(game_wrapper._level_progress_max, self.real_x_pos) #Quitamos las conexiones con el wrapper debido a problemas en windows #self.world = game_wrapper.world #Quitamos las conexiones con el wrapper debido a problemas en windows
        world_level = pyboy.get_memory_value(0xFFB4)
        world = world_level >> 4
        level = world_level & 0x0F

        self.world = (world, level)
        self._level_progress_max = self.real_x_pos


class MarioAI(AISettingsInterface):
    def __init__(self):
        self.realMax = [] #[[1,1, 2500], [1,1, 200]]		

    def GetReward(self, prevGameState: GameState, pyboy):
        """
        previousMario = mario before step is taken
        current_mario = mario after step is taken
        """
        timeRespawn = pyboy.get_memory_value(0xFFA6) #Time until respawn from death (set when Mario has fell to the bottom of the screen) 
        if(timeRespawn > 0): # if we cant move return 0 reward otherwise we could be punished for crossing a level
            return 0

        "Get current game state"
        current_mario = self.GetGameState(pyboy)

        if max((current_mario.world[0] - prevGameState.world[0]), (current_mario.world[1] - prevGameState.world[1])): # reset level progress max
            # reset level progress max on new level
            for _ in range(0,5):
                pyboy.tick() # skip frames to get updated x pos on next level

            current_mario = self.GetGameState(pyboy)

            pyboy.game_wrapper()._level_progress_max = current_mario.real_x_pos
            current_mario._level_progress_max = current_mario.real_x_pos


        if len(self.realMax) == 0:
            self.realMax.append([current_mario.world[0], current_mario.world[1], current_mario._level_progress_max])
        else:
            r = False
            for elem in self.realMax: # fix max length
                if elem[0] == current_mario.world[0] and elem[1] == current_mario.world[1]:
                    elem[2] = current_mario._level_progress_max
                    r = True
                    break # leave loop
            
            
            if r == False: # this means this level does not exist
                self.realMax.append([current_mario.world[0], current_mario.world[1], current_mario._level_progress_max])
            

        # reward function simple
        clock = current_mario.time_left - prevGameState.time_left
        movement = current_mario.real_x_pos - prevGameState.real_x_pos
        death = -15*(current_mario.lives_left - prevGameState.lives_left)
        levelReward = 15*max((current_mario.world[0] - prevGameState.world[0]), (current_mario.world[1] - prevGameState.world[1])) # +15 if either new level or new world

        reward = clock + death + movement + levelReward

        return reward

    def GetActions(self):
        baseActions = [WindowEvent.PRESS_ARROW_RIGHT,
                        WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT]

        totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
        withoutRepeats = []

        for combination in totalActionsWithRepeats:
            reversedCombination = combination[::-1]
            if(reversedCombination not in withoutRepeats):
                withoutRepeats.append(combination)

        filteredActions = [[action] for action in baseActions] + withoutRepeats

        # remove  ['PRESS_ARROW_RIGHT', 'PRESS_ARROW_LEFT']
        del filteredActions[4]

        return filteredActions

    def PrintGameState(self, pyboy):
        gameState = GameState(pyboy)
        game_wrapper = pyboy.game_wrapper()

        print("'Fake', level_progress: ", game_wrapper.level_progress)
        print("'Real', level_progress: ", gameState.real_x_pos)
        print("_level_progress_max: ", gameState._level_progress_max)
        print("World: ", gameState.world)
        print("Time respawn", pyboy.get_memory_value(0xFFA6))

    def GetGameState(self, pyboy):
        return GameState(pyboy)

    def GetHyperParameters(self) -> Config:
        config = Config()
        #config.exploration_rate_decay = 0.999
        config.exploration_rate_decay = 0.9995 #Cambio por Nacho para reducir la tasa de decaimiento de la tasa de exploración, lo que hace que el agente explore durante más tiempo.
        return config

    def GetLength(self, pyboy):
        result = sum([x[2] for x in self.realMax])

        self.realMax = []

        return result

    def IsDone(self, pyboy): #Quitamos las conexiones con el wrapper debido a problemas en windows

        lives = pyboy.get_memory_value(0xDA15)

        return lives == 0
