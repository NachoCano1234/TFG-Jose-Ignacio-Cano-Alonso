from collections import defaultdict, deque
import random
#import sys
import numpy as np
import torch
from AISettings.AISettingsInterface import Config
from model import MetaDDQN

#based on pytorch RL tutorial by yfeng997: https://github.com/yfeng997/MadMario/blob/master/agent.py
class AIPlayer:
    def __init__(self, state_dim, action_space_dim, save_dir, date, config: Config):
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.save_dir = save_dir
        self.date = date
        self.device = "cpu"

        '''
        if torch.cuda.is_available():
            #self.device = "cuda"
            self.device = "cpu" #Cambio por Nacho para forzar el uso de CPU, ya que el estado es un LazyFrame y no se puede convertir directamente a tensor sin copiar los datos, lo que hace que el uso de GPU no sea eficiente en este caso.
        ''' #Cambio por Nacho MetaRL

        #self.net = DDQN(self.state_dim, self.action_space_dim).to(device=self.device)
        self.net = MetaDDQN(self.state_dim, self.action_space_dim).to(device=self.device) #Cambio por Nacho MetaRL para utilizar la red neuronal MetaDDQN, que es una red neuronal que puede aprender a aprender, lo que permite al agente adaptarse a nuevas tareas de manera más eficiente.

        self.config = config

        self.exploration_rate = self.config.exploration_rate 
        self.exploration_rate_decay = self.config.exploration_rate_decay
        self.exploration_rate_min = self.config.exploration_rate_min
        self.curr_step = 0

        """
            Memory
        """
        #self.memory = deque(maxlen=self.config.deque_size) #Este es un buffer global, lo que mezcla tareas y por tanto impide aislar las experiencias
        self.task_buffers = defaultdict(lambda: deque(maxlen=self.config.deque_size)) #Cambio por Nacho MetaRL. Este buffer divide la experiencia del agente en tareas. El buffer se compone de varias tareas, y cada tarea se guarda sus propias experiencias
        self.batch_size = self.config.batch_size
        self.context_size = 32 #Cambio por Nacho MetaRL. Este es el número de experiencias que se utilizan para inferir el contexto de la tarea actual.
        self.meta_batch_size = 1 #4 #Cambio por Nacho MetaRL. Este es el número de tareas que se utilizan para actualizar la red neuronal.
        self.save_every = self.config.save_every  # no. of experiences between saving Mario Net

        """
            Q learning
        """
        self.gamma = self.config.gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.learning_rate_decay)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = self.config.burnin  # min. experiences before training
        self.learn_every = self.config.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = self.config.sync_every  # no. of experiences between Q_target & Q_online sync

        self.current_z = torch.zeros((1, 64)).to(self.device) #Cambio por Nacho MetaRL para inicializar el contexto inferido de la tarea actual como un vector de ceros

        self.valid_actions_per_task = { #Cambio por Nacho MetaRL para arreglar mezcla de acciones inválidas
            "Super_Mario_Land": 6,
            "Kirby_Dream_Land": 21
        }

    def act(self, state, task_id): #Cambio por Nacho MetaRL para añadir el campo task_id, que permite al agente seleccionar la acción adecuada para la tarea actual, lo que permite al agente adaptarse a nuevas tareas de manera más eficiente.
        """
            Given a state, choose an epsilon-greedy action and update value of step.

            Inputs:
            state(LazyFrame): A single observation of the current state, dimension is (state_dim)
            Outputs:
            action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if (random.random() < self.exploration_rate):
            #actionIdx = random.randint(0, self.action_space_dim-1)
            valid_actions = self.valid_actions_per_task[task_id] #Cambio por Nacho MetaRL para arreglar mezcla de acciones inválidas
            action = random.randint(0, valid_actions - 1) #Cambio por Nacho MetaRL para arreglar mezcla de acciones inválidas
        # EXPLOIT
        else:
            #state = np.array(state)
            #state = torch.tensor(state).float().to(device=self.device)
            state = np.array(state, copy=False) #Cambio por Nacho para evitar la copia innecesaria de datos
            
            #state = state.unsqueeze(0)
            state = torch.tensor(state).float().to(self.device).unsqueeze(0) #Cambio por Nacho para convertir el estado a tensor y agregar una dimensión de batch

            '''
            with torch.no_grad(): 
                neuralNetOutput = self.net(state, model="online") #Cambio por Nacho para obtener la salida de la red neuronal utilizando el modelo "online", ya que el estado es un LazyFrame y no se puede convertir directamente a tensor sin copiar los datos.
            ''' #Cambio por Nacho MetaRL. Se elimina el bloque original de selección de acción, ya que ahora se utiliza el contexto inferido de la tarea actual para obtener los valores Q, lo que permite al agente adaptarse a nuevas tareas de manera más eficiente.
            
            #neuralNetOutput = self.net(state, model="online")
            
            z = self.current_z
            z = z.expand(1, -1) #Cambio por Nacho MetaRL para expandir el contexto inferido de la tarea actual a lo largo de la dimensión de batch

            q_values = self.net(state, z) #Cambio por Nacho MetaRL para obtener los valores Q utilizando el contexto inferido de la tarea actual, lo que permite al agente adaptarse a nuevas tareas de manera más eficiente.
            
            #START Nacho #Cambio por Nacho MetaRL para arreglar mezcla de acciones inválidas
            valid_actions = self.valid_actions_per_task[task_id]

            mask = torch.full_like(q_values, -1e9)
            mask[:, :valid_actions] = 0

            q_values = q_values + mask
            #END Nacho

            #actionIdx = torch.argmax(neuralNetOutput, axis=1).item()
            action = torch.argmax(q_values, axis=1).item() #Cambio por Nacho MetaRL para seleccionar la acción con el valor Q más alto

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action

    def cache(self, state, next_state, action, reward, done, task_id): #Cambio por Nacho MetaRLLe añadimos el campo task_id
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool)
        """
        '''
        state = np.array(state)
        next_state = np.array(next_state)

        state = torch.tensor(state).float().to(device=self.device)
        next_state = torch.tensor(next_state).float().to(device=self.device)
        action = torch.tensor([action]).to(device=self.device)
        reward = torch.tensor([reward]).to(device=self.device)
        done = torch.tensor([done]).to(device=self.device)
        '''
        state = np.array(state, copy=False) #Cambio por Nacho para evitar la copia innecesaria de datos
        next_state = np.array(next_state, copy=False) #Cambio por Nacho para evitar la copia innecesaria de datos

        self.task_buffers[task_id].append((state, next_state, action, reward, done)) #Cambio por Nacho MetaRL

    '''
    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        #Empieza el cambio por Nacho para evitar la copia innecesaria de datos
        state, next_state, action, reward, done = zip(*batch)

        state = torch.from_numpy(np.array(state)).float().to(self.device)
        next_state = torch.tensor(np.array(next_state)).float().to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        done = torch.tensor(done).to(self.device)

        return state, next_state, action, reward, done
        #Fin del cambio por Nacho para evitar la copia innecesaria de datos
        #state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        #return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    ''' #Cambio por Nacho MetaRL. Se elimina el método recall original, ya que ahora se utiliza un método de muestreo específico para cada tarea.

    def sample_context(self, task_id): #Cambio por Nacho MetaRL para añadir el método sample_context al completo, que permite al agente inferir el contexto de la tarea actual utilizando las experiencias almacenadas en el buffer de la tarea.

        # Si no hay suficientes datos, devolver contexto neutro
        buffer = self.task_buffers[task_id]
        if len(buffer) < self.context_size:
            return torch.zeros((1, 64)).to(self.device) #Cambio por Nacho MetaRL para devolver un vector de ceros si el buffer de la tarea no tiene suficientes experiencias para inferir el contexto.

        # Samplear transiciones

        batch = random.sample(buffer, self.context_size)
        states, _, actions, rewards, _ = zip(*batch)

        # Convertir a tensores
        '''
        actions = torch.tensor(actions).long().to(self.device)
        
        actions = torch.tensor(actions).float().unsqueeze(1).to(self.device)
        actions = torch.nn.functional.one_hot(actions, num_classes=self.action_space_dim).float()
        ''' #Cambio por Nacho MetaRL para convertir las acciones a tensores de una sola dimensión y luego a one-hot

        actions = torch.tensor(actions).long().to(self.device)
        actions = torch.nn.functional.one_hot(actions, num_classes=self.action_space_dim).float()

        rewards = torch.tensor(rewards).float().unsqueeze(1).to(self.device)

        # Extraer features del estado (CNN)
        states = torch.tensor(np.array(states)).float().to(self.device) #Cambio por Nacho MetaRL para convertir los estados a tensores.

        with torch.no_grad():
            features = self.net.feature(states)

        # Concatenar: [features, action, reward]
        context_input = torch.cat([features, actions, rewards], dim=1)

        # Pasar por encoder
        z = self.net.context_encoder(context_input)

        # Agregación (mean pooling)
        z = z.mean(dim=0, keepdim=True)

        return z
    
    def sample_batch(self, task_id): #Cambio por Nacho MetaRL para añadir el método sample_batch al completo, el cual permite al agente muestrear un lote de experiencias específico para la tarea actual.
        buffer = self.task_buffers[task_id]

        batch = random.sample(buffer, self.batch_size)
        state, next_state, action, reward, done = zip(*batch)

        state = torch.tensor(np.array(state)).float().to(self.device)
        next_state = torch.tensor(np.array(next_state)).float().to(self.device)
        action = torch.tensor(action).long().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)

        return state, next_state, action, reward, done

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        if len(self.task_buffers) < self.meta_batch_size: #Cambio por Nacho MetaRL para asegurarse de que hay suficientes tareas con experiencias en el buffer antes de intentar aprender.
            return None, None
        
        '''
        # Sample from memory get self.batch_size number of memories
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate, make predictions for the each memory
        td_est = self.td_estimate(state, action)

        # Get TD Target make predictions for next state of each memory
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)
        ''' #Cambio por Nacho MetaRL. Se elimina el bloque original de aprendizaje, ya que ahora se utiliza un método de muestreo específico para cada tarea.

        #START Cambio por Nacho MetaRL para añadir el bloque de aprendizaje específico para cada tarea.
        tasks = random.sample(list(self.task_buffers.keys()), self.meta_batch_size)

        total_q = 0
        total_loss = 0
        valid_tasks = 0

        for task_id in tasks:

            if len(self.task_buffers[task_id]) < self.batch_size:
                continue

            state, next_state, action, reward, done = self.sample_batch(task_id)
            z = self.sample_context(task_id).detach()
            z = z.expand(state.shape[0], -1)

            # Q(s, z)
            q_values = self.net(state, z)
            batch_indices = torch.arange(state.shape[0], device=self.device)

            current_Q = q_values[batch_indices, action]

            # Target
            with torch.no_grad():
                next_q = self.net(next_state, z)
                best_action = torch.argmax(next_q, axis=1)

                target_q = self.net(next_state, z, model="target")[batch_indices, best_action]

                td_target = reward + (1 - done) * self.gamma * target_q

            loss = self.loss_fn(current_Q, td_target)

            total_q += current_Q.mean()
            total_loss += loss
            valid_tasks += 1

        if valid_tasks == 0:
            return None, None

        total_loss /= valid_tasks
        total_q /= valid_tasks

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return total_q.item(), total_loss.item()
        #return None, total_loss.item()
        #END Cambio por Nacho MetaRL

    '''
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step() #

        return loss.item()
    '''  #Cambio por Nacho MetaRL. Se elimina el método update_Q_online original, ya que ahora se utiliza un método de muestreo específico para cada tarea

    def start_episode(self, task_id): #Cambio por Nacho MetaRL para añadir el método start_episode, que se llama al inicio de cada episodio para establecer el contexto de la tarea actual.
        self.current_z = self.sample_context(task_id)

    def sync_Q_target(self):
        #self.net.target.load_state_dict(self.net.online.state_dict())
        self.net.target.load_state_dict(self.net.q_net.state_dict()) #Cambio por Nacho MetaRL. Al ya no haber self.net.online, sino que la red principal se llama self.net.q_net, se actualiza el método para sincronizar la red "target" con la red "q_net".

    '''
    def td_estimate(self, state, action):
        """
            Output is batch_size number of rewards = Q_online(s,a) * 32
        """
        modelOutPut = self.net(state, model="online")
        current_Q = modelOutPut[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_Q
    ''' #Cambio por Nacho MetaRL. Se elimina el método td_estimate original, ya que ahora se utiliza un método de muestreo específico para cada tarea

    '''
    @torch.no_grad()
    def td_target(self, reward, next_state, done, z): #Cambio por Nacho MetaRL para añadir el campo z, que es el contexto inferido de la tarea actual, lo que permite al agente adaptarse a nuevas tareas de manera más eficiente.
        """
            Output is batch_size number of Q*(s,a) = r + (1-done) * gamma * Q_target(s', argmax_a'( Q_online(s',a') ) )
        """
        #next_state_Q = self.net(next_state, model="online") 
        next_q = self.net(next_state, z) #Cambio por Nacho MetaRL para obtener los valores Q del siguiente estado utilizando el contexto inferido de la tarea actual, lo que permite al agente adaptarse a nuevas tareas de manera más eficiente.
        best_action = torch.argmax(next_q, axis=1) #Cambio por Nacho MetaRL para seleccionar la acción con el valor Q más alto en el siguiente estado, utilizando el contexto inferido de la tarea actual, lo que permite al agente adaptarse a nuevas tareas de manera más eficiente.
        #next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action] # Q_target(s', argmax_a'( Q_online(s',a') ) )
        target_q = self.net(next_state, z, model="target")[np.arange(next_state.shape[0]), best_action] #Cambio por Nacho MetaRL para obtener el valor Q del siguiente estado utilizando el modelo "target" y el contexto inferido de la tarea actual, lo que permite al agente adaptarse a nuevas tareas de manera más eficiente.
        #return (reward + (1 - done.float()) * self.gamma * next_Q).float() # Q*(s,a)
        return (reward + (1 - done) * self.gamma * target_q).float() #Cambio por Nacho MetaRL para calcular el valor objetivo de TD utilizando el contexto inferido de la tarea actual, lo que permite al agente adaptarse a nuevas tareas de manera más eficiente.
    '''
    def loadModel(self, path):
        dt = torch.load(path, map_location=torch.device(self.device))
        self.net.load_state_dict(dt["model"])
        self.exploration_rate = dt["exploration_rate"]
        print(f"Loading model at {path} with exploration rate {self.exploration_rate}")

    def saveHyperParameters(self):
        save_HyperParameters = self.save_dir / "hyperparameters"
        with open(save_HyperParameters, "w") as f:
            f.write(f"exploration_rate = {self.config.exploration_rate}\n")
            f.write(f"exploration_rate_decay = {self.config.exploration_rate_decay}\n")
            f.write(f"exploration_rate_min = {self.config.exploration_rate_min}\n")
            f.write(f"deque_size = {self.config.deque_size}\n")
            f.write(f"batch_size = {self.config.batch_size}\n")
            f.write(f"gamma (discount parameter) = {self.config.gamma}\n")
            f.write(f"learning_rate = {self.config.learning_rate}\n")
            f.write(f"learning_rate_decay = {self.config.learning_rate_decay}\n")
            f.write(f"burnin = {self.config.burnin}\n")
            f.write(f"learn_every = {self.config.learn_every}\n")
            f.write(f"sync_every = {self.config.sync_every}")

    def save(self):
        """
            Save the state to directory
        """
        save_path = (self.save_dir / f"mario_net_0{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
