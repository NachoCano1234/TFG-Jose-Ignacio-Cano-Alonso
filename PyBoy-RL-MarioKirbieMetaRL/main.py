import datetime
import random
from pathlib import Path
from pyboy.pyboy import *
from gym.wrappers import FrameStack, NormalizeObservation
from AISettings.AISettingsInterface import AISettingsInterface, Config
from AISettings.MarioAISettings import MarioAI
from AISettings.KirbyAISettings import KirbyAI
from MetricLogger import MetricLogger
from agent import AIPlayer
from wrappers import SkipFrame, ResizeObservation
import sys
from CustomPyBoyGym import CustomPyBoyGym
from functions import alphanum_key

"""
  Variables
"""
episodes = 400 #40000
# gym variables  documentation: https://docs.pyboy.dk/openai_gym.html#pyboy.openai_gym.PyBoyGymEnv
observation_types = ["raw", "tiles", "compressed", "minimal"]
observation_type = observation_types[1]
action_types = ["press", "toggle", "all"]
action_type = action_types[0]
#gameDimentions = (20, 16)
#frameStack = 4
gameDimentions = (16, 12) #Cambio por Nacho para reducir las dimensiones del juego
frameStack = 3 #Cambio por Nacho para reducir el tamaño del frame stack
quiet = False
train = False
playtest = False

"""
  Choose game
"""
gamesFolder = Path("games")
games = [os.path.join(gamesFolder, f) for f in os.listdir(gamesFolder) if (os.path.isfile(os.path.join(gamesFolder, f)) and f.endswith(".gb"))]
gameNames = [f.replace(".gb", "") for f in os.listdir(gamesFolder) if (os.path.isfile(os.path.join(gamesFolder, f)) and f.endswith(".gb"))]

'''
print("Avaliable games: ", games)
for cnt, gameName in enumerate(games, 1):
	sys.stdout.write("[%d] %s\n\r" % (cnt, gameName))

choice = int(input("Select game[1-%s]: " % cnt)) - 1
game = games[choice]
gameName = gameNames[choice]
'''
"""
  Choose mode
"""
modes = ["Evaluate (HEADLESS)", "Evaluate (UI)",
		 "Train (HEADLESS)", "Train (UI)", "Playtest (UI)"]
for cnt, modeName in enumerate(modes, 1):
	sys.stdout.write("[%d] %s\n\r" % (cnt, modeName))

mode = int(input("Select mode[1-%s]: " % cnt)) - 1

if mode == 0:
	quiet = True
	train = False
elif mode == 1:
	quiet = False
	train = False
elif mode == 2:
	quiet = True
	train = True
elif mode == 3:
	quiet = False
	train = True
elif mode == 4:
	quiet = False
	playtest = True

"""
  Logger
"""
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
#save_dir es la carpeta donde se guarda la info del entrenamiento actual , y checkpoint_dir es la carpeta donde se guardan todos los entrenamientos
save_dir = Path("checkpoints") / "MetaRL_MultiGame" / now #Cambio por Nacho MetaRL. Al tener ahora dos juegos al mismo tiempo, hay que saber diferenciarlos para poder actualizar sus checkpoints
checkpoint_dir = Path("checkpoints") / "MetaRL_MultiGame" #Cambio por Nacho MetaRL. Ahora los checkpoints se guardan en una carpeta común
#checkpoint_dir = Path("checkpoints") / gameName
save_dir_eval = Path("checkpoints") / "MetaRL_MultiGame" / (now + "-eval") #Cambio por Nacho MetaRL
#save_dir_eval = Path("checkpoints") / gameName / (now + "-eval")
#save_dir_boss = Path("checkpoints") / gameName / (now + "-boss") #Cambio en Nacho MetaRL. Eliminamos todo lo relacionado con boss model para dejar solo un modelo

"""
  Load emulator
"""
#pyboy = PyBoy(game, window_type="headless" if quiet else "SDL2", window_scale=3, debug=False, game_wrapper=True) #Cambio por Nacho MetaRL. Al no existir el pyboy global, esto ya se elimina

"""
  Load enviroment
"""
'''
aiSettings = AISettingsInterface()
if pyboy.game_wrapper().cartridge_title == "SUPER MARIOLAN":
	aiSettings = MarioAI()
if pyboy.game_wrapper().cartridge_title == "KIRBY DREAM LA":
	aiSettings = KirbyAI()

env = CustomPyBoyGym(pyboy, observation_type=observation_type)
env.setAISettings(aiSettings)  # use this settings
filteredActions = aiSettings.GetActions()  # get possible actions
print("Possible actions: ", [[WindowEvent(i).__str__() for i in x] for x in filteredActions])
''' #Cambio por Nacho MetaRL. Al no existir el pyboy global, esto ya se elimina

"""
  Apply wrappers to enviroment
"""
'''
#env = SkipFrame(env, skip=4)
env = SkipFrame(env, skip=6) #Cambio por Nacho para aumentar el número de frames que se saltan
env = ResizeObservation(env, gameDimentions)  # transform MultiDiscreate to Box for framestack
env = NormalizeObservation(env)  # normalize the values
env = FrameStack(env, num_stack=frameStack)
''' #Cambio por Nacho MetaRL. Al no existir el env global, esto ya se elimina

"""
  Load AI players
"""

#aiPlayer = AIPlayer((frameStack,) + gameDimentions, len(filteredActions), save_dir, now, aiSettings.GetHyperParameters())
#bossAiPlayer = AIPlayer((frameStack,) + gameDimentions, len(filteredActions), save_dir_boss, now, aiSettings.GetBossHyperParameters()) 
#Cambio por Nacho MetaRL. Al quitar el bossAiPlayer ya no se usa un modelo distinto según la tarea, debido a que en caso contrario no sería MetaRL

resume = False   # pon False si quieres empezar desde cero sin usar checkpoints (por ejemplo para empezar a entrenar un nuevo modelo), y True para el caso contrario

if resume:
    checkpoint_dir = Path("checkpoints") / gameName

    folderList = [name for name in os.listdir(checkpoint_dir)
                  if os.path.isdir(checkpoint_dir / name)
                  and len(os.listdir(checkpoint_dir / name)) != 0]

    if len(folderList) > 0:
        print("Available checkpoints:")
        for cnt, fileName in enumerate(folderList, 1):
            print(f"[{cnt}] {fileName}")

        choice = int(input("Select folder to resume training: ")) - 1
        folder = folderList[choice]

        fileList = [f for f in os.listdir(checkpoint_dir / folder) if f.endswith(".chkpt")]
        fileList.sort(key=alphanum_key)

        modelPath = checkpoint_dir / folder / fileList[-1]

        print("Loading checkpoint:", modelPath)

        aiPlayer.loadModel(modelPath)
        #bossAiPlayer.loadModel(modelPath) #Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer
#END Nacho Cano

if mode < 2:  # evaluate
	# load model
	folderList = [name for name in os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir / name) and len(os.listdir(checkpoint_dir / name)) != 0]

	if len(folderList) == 0:
		print("No models to load in path: ", save_dir)
		quit()

	for cnt, fileName in enumerate(folderList, 1):
		sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))

	choice = int(input("Select folder with platformer model[1-%s]: " % cnt)) - 1
	folder = folderList[choice]
	print(folder)

	fileList = [f for f in os.listdir(checkpoint_dir / folder) if f.endswith(".chkpt")]
	fileList.sort(key=alphanum_key)
	if len(fileList) == 0:
		print("No models to load in path: ", folder)
		quit()

	modelPath = checkpoint_dir / folder / fileList[-1]
	aiPlayer.loadModel(modelPath)

	'''
	choice = int(input("Select folder with boss model[1-%s] (if not using boss model select same as previous): " % cnt)) - 1
	folder = folderList[choice]
	print(folder)

	fileList = [f for f in os.listdir(checkpoint_dir / folder) if f.endswith(".chkpt")]
	fileList.sort(key=alphanum_key)
	if len(fileList) == 0:
		print("No models to load in path: ", folder)
		quit()

	bossModelPath = checkpoint_dir / folder / fileList[-1]
	''' #Cambio por Nacho MetaRL. Al no haber modelo boss ya, este codigo ya no hace nada
	#bossAiPlayer.loadModel(bossModelPath) #Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer

"""
  Main loop
"""

if train:
	#pyboy.set_emulation_speed(0) #Cambio por Nacho MetaRL. Al no existir el pyboy global, esto ya se elimina
	
	save_dir.mkdir(parents=True)
	logger = MetricLogger(save_dir) #Añadido en Nacho MetaRL. Ahora al usar un solo modelo tenemos que pasar este a metricLogger en lugar del modelo boss que usabamos antes
	
	#save_dir_boss.mkdir(parents=True) #Cambio en Nacho MetaRL. Eliminamos todo lo relacionado con boss model para dejar solo un modelo
	#logger = MetricLogger(save_dir_boss) #Cambio en Nacho MetaRL. Eliminamos todo lo relacionado con boss model para dejar solo un modelo
	
	#bossAiPlayer.saveHyperParameters() #Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer
	#bossAiPlayer.net.train() #Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer

	'''
	aiPlayer.saveHyperParameters()
	aiPlayer.net.train()
	player = aiPlayer
	''' #Cambio por Nacho MetaRL. Al no existir el aiPlayer global, esto ya se elimina

	print("Training mode")
	print("Total Episodes: ", episodes)

	envs = {}

	for i, game in enumerate(games):
		pyboy = PyBoy(game, window_type="headless" if quiet else "SDL2", window_scale=3, debug=False, game_wrapper=True)
		
		'''
		if pyboy.game_wrapper().cartridge_title == "SUPER MARIOLAN":
			aiSettings = MarioAI()
		else:
			aiSettings = KirbyAI()
		'''

		#START Nacho. Quitamos las conexiones con el wrapper debido a problemas en windows
		if "Mario" in gameNames[i]:
			aiSettings = MarioAI()
		else:
			aiSettings = KirbyAI()
		#END Nacho

		env = CustomPyBoyGym(pyboy, observation_type=observation_type)
		env.setAISettings(aiSettings)

		env = SkipFrame(env, skip=6)
		env = ResizeObservation(env, gameDimentions)
		env = NormalizeObservation(env)
		env = FrameStack(env, num_stack=frameStack)
		filteredActions = aiSettings.GetActions()

		envs[gameNames[i]] = (env, aiSettings, filteredActions)

	for env, _, _ in envs.values(): 
		env.pyboy.set_emulation_speed(0) 
	
	config = Config() #Cambio por Nacho MetaRL. Creamos un config comun a ambos juegos
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
	
	action_space_dim = max(len(v[2]) for v in envs.values()) #Cambio por Nacho MetaRL. Al tener varios juegos con diferentes espacios de acción, se toma el espacio de acción máximo entre todos los juegos para crear un solo modelo que pueda manejar todos los juegos, aunque luego en cada paso se le indicará al modelo cuales son las acciones válidas para el juego actual a través del campo task_id.
	aiPlayer = AIPlayer((frameStack,) + gameDimentions, action_space_dim, save_dir, now, config) #Cambio por Nacho MetaRL.
	aiPlayer.saveHyperParameters()
	aiPlayer.net.train()
	player = aiPlayer

	for e in range(episodes):
		game_idx = random.randint(0, len(games) - 1)
		gameName = gameNames[game_idx]

		env, aiSettings, filteredActions = envs[gameName]

		observation = env.reset()
		task_id = gameName #Cambio por Nacho MetaRL. Se pone fuera del while para que no se recalcule en cada episodio (pregunta a chatgpt que hace)
		player.start_episode(task_id)
		
		start = time.time()
		steps = 0 #Creado por Nacho

		while True:
			'''
			if aiSettings.IsBossActive(pyboy):
				player = bossAiPlayer
			else:
				player = aiPlayer
			''' #Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer
			player = aiPlayer
			actionId = player.act(observation, task_id) #Cambio por Nacho MetaRL. Le pasamos el task_id al agente
			actionId = min(actionId, len(filteredActions) - 1) #Cambio por Nacho MetaRL para evitar que el modelo elija una acción que no existe en el juego actual
			actions = filteredActions[actionId]
			# Agent performs action and moves 1 frame
			next_observation, reward, done, info = env.step(actions)
			steps += 1 #Creado por Nacho

			# Remember
			player.cache(observation, next_observation, actionId, reward, done, task_id) #Cambio por Nacho MetaRL. Se ha añadido el campo task_id
			# Learn
			#q, loss = player.learn()
			if steps % 10 == 0:
				q, loss = player.learn() #Cambio por Nacho MetaRL. Hacemos que el modelo aprenda cada 10 pasos en vez de en cada paso para que no haya demasiado ruido ni un contexto muy inestable
			else:
				q, loss = None, None
			# Logging
			logger.log_step(reward, loss, q, player.scheduler.get_last_lr())
			# Update state
			observation = next_observation

			if done or time.time() - start > 200: #Creado por Nacho para evitar que los episodios duren demasiado tiempo, ya que a veces el agente puede quedarse atascado en una parte del juego y no avanzar, lo que hace que el episodio dure mucho tiempo sin obtener recompensas ni aprender nada nuevo.
				break

		logger.log_episode()
		aiPlayer.exploration_rate *= 0.98 #Cambio por Nacho para reducir la tasa de decaimiento de la tasa de exploración, lo que hace que el agente explore durante más tiempo.
		#Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer
		#bossAiPlayer.exploration_rate *= 0.98 
		#logger.record(episode=e, epsilon=player.exploration_rate, stepsThisEpisode=player.curr_step, maxLength=aiSettings.GetLength(pyboy))
		logger.record(
				episode=e,
				epsilon=player.exploration_rate,
				stepsThisEpisode=steps,
				maxLength=aiSettings.GetLength(pyboy),
				gameName=gameName
			)
		if e % 10 == 0:
			aiPlayer.save()
			#bossAiPlayer.save()	#Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer

	aiPlayer.save()
	#bossAiPlayer.save() #Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer
	for env, _, _ in envs.values():
		env.close() #Cambio por Nacho MetaRL. Como ahora no hay solo un entorno, tenemos que cerrar todos los entornos al finalizar el entrenamiento
elif not train and not playtest:
	print("Evaluation mode")
	pyboy.set_emulation_speed(1)

	save_dir_eval.mkdir(parents=True)
	logger = MetricLogger(save_dir_eval)

	aiPlayer.exploration_rate = 0
	aiPlayer.net.eval()

	#bossAiPlayer.exploration_rate = 0 #Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer
	#bossAiPlayer.net.eval() #Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer

	player = aiPlayer
	for e in range(episodes):
		observation = env.reset()
		
		task_id = pyboy.game_wrapper().cartridge_title #Cambio por Nacho MetaRL. Se pone fuera del while para que no se recalcule en cada episodio (pregunta a chatgpt que hace)
		player.start_episode(task_id)
		#task_id = (aiSettings.IsBossActive(pyboy), pyboy.game_wrapper().world) 
		steps = 0
		
		while True:
			'''
			if aiSettings.IsBossActive(pyboy):
				player = bossAiPlayer
			else:
				player = aiPlayer
			''' #Cambio por Nacho MetaRL. Quitamos todos los bossAiPlayer
			#actionId = player.act(observation)
			actionId = player.act(observation, task_id) #Cambio por Nacho MetaRL. Le pasamos el task_id al agente
			action = filteredActions[actionId]
			next_observation, reward, done, info = env.step(action)

			logger.log_step(reward, 1, 1, 1)
			steps += 1 #Creado por Nacho

			print("Episode running... steps:", steps) #Creado por Nacho. Nos ayuda a ver cuantos pasos está durando el episodio	
			print("Reward: ", reward)
			print("Action: ", [WindowEvent(i).__str__() for i in action])
			aiSettings.PrintGameState(pyboy)

			observation = next_observation

			# print(reward)
			if done:
				break

		logger.log_episode()
		logger.record(episode=e, epsilon=player.exploration_rate, stepsThisEpisode=player.curr_step, maxLength=aiSettings.GetLength(pyboy), gameName=gameName)
	for env, _, _ in envs.values():
		env.close() #Cambio por Nacho MetaRL. Como ahora no hay solo un entorno, tenemos que cerrar todos los entornos al finalizar la evaluación

elif playtest:
	pyboy.set_emulation_speed(1)
	env.reset()
	print("Playtest mode")
	while True:
		previousGameState = aiSettings.GetGameState(pyboy)
		env.pyboy.tick()

		print("Reward: ", aiSettings.GetReward(previousGameState, pyboy))
		print("Real max length: ", aiSettings.GetLength(pyboy))
		aiSettings.PrintGameState(pyboy)

		if env.game_wrapper.game_over():
			break
