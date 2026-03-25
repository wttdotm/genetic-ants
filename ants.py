#!/usr/local/bin/python3

'''
This script creates a number of ants, which gradually evolve to survive and thrive in their environment.
The ants move, eat, mate, produce offspring, and die in an environment containing food and obstacles.

Ant movement:
any combo of dx=1, -1 and dy=1, -1

Ant actions:
5		- propose mating with a colliding ant. If that ant also proposes mating, then they produce offspring
  		  with a random combination of their genes
6		- eat a colliding food
7		- die

todo:
8		- attack a colliding ant. The ants' genes determines who wins the fight and kills the other.
9		- befriend a colliding ant. The ants then share food (todo--determine this mechanism) and protect each other

Ant inputs:
- colliding objects (ants, food, obstacles)
- grid location
- internal state (timers, etc)
'''

import pygame
import random, math
import numpy as np

BLACK = (0,		0, 		0)
GREY =	(150,	150,	150)
WHITE = (255, 	255, 	255)
RED =	(255,	0,		0)

INITIAL_ANTS = 30
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 400
FPS = 100
STEP_SIZE = 1
INTERACTION_PERIOD = 200	#period in turns
INTERACTION_DELAY = 200 #delay until ant matures enough to mate or fight with other ants

# todos:
# influence actions with direct genes
# introduce fixed mutation rate in mating
# genetically influence randomness in movement

class Ant(pygame.sprite.Sprite):

	##
	## @brief      Constructs the ant
	##
	## @param      self    The object
	## @param      agent   The agent controlling the ant
	## @param      game    The game object describing various game state items
	## @param      init_x  The initial x position of the ant
	## @param      init_y  The initial y position of the ant
	## @param      width   The width of the ant
	## @param      height  The height of the ant on screen
	##
	def __init__(self, agent, game, init_x, init_y, width=15, height=15):

		super().__init__()
		self.agent = agent

		#make the generic square that represents this ant
		self.image = pygame.Surface((width, height))
		print(agent.get_color())
		self.image.fill(agent.get_color())

		#initialize the object position
		self.rect = self.image.get_rect()
		self.rect.x = init_x
		self.rect.y = init_y

		#initialize the game object
		self.game = game

		#add the ant to the relevant object lists
		self.id = game.ant_count
		game.ant_count += 1
		print('Ant {} spawning at ({},{}).'.format(self.id, init_x, init_y))
		self.game.ant_list.add(self)
		self.game.all_sprites_list.add(self)

		#initialize previous movement
		self.prev_dx = 0
		self.prev_dy = 0

		#prev action
		self.prev_action = 0

		#ant's can't mate til they're a bit older
		self.prev_interaction = INTERACTION_DELAY

		#ants get boobs
		self.boobs = True

	##
	## @brief      Called once per tick to perform an action and update the
	##             ant's location on screen
	##
	## @param      self  The object
	##
	## @return     none
	##
	def update(self):
		self.agent.act(self)

	##
	## @brief      Interact with another ant with whom you've collided
	##
	## @param      self       The object
	## @param      other_ant  The other ant
	##
	## @return     none
	##
	def interact(self, other_ant):
		#ants can only interact every so often
		if self.game.turn_count - self.prev_interaction > INTERACTION_PERIOD and \
				self.game.turn_count - other_ant.prev_interaction > INTERACTION_PERIOD:
				#but when they do, they either fight or mate
				f_interact = Ant.interactions[self.agent.select_interaction(other_ant.agent.genes)]
				f_interact(self, other_ant)

	##
	## @brief      Attempt to mate with another ant. Does nothing if the mating fails.
	##
	## @param      self       The object
	## @param      other_ant  The other ant
	##
	## @return     None
	##
	def attempt_mate(self, other_ant):
		child_genes = self.agent.mate(other_ant.agent.genes)
		#if the mating produced a viable offspring (otherwise is None), make a new ant and spawn it nearby
		if child_genes is None:
			return
		else:
			child_spawn_angle = random.uniform(0, math.pi)
			child = Ant(Agent(child_genes), self.game,
				#spawn the child 20 units away at some random angle
				self.rect.x + 20*math.sin(child_spawn_angle), 
				self.rect.y + 20*math.cos(child_spawn_angle))

			#reset the mating timer
			self.prev_interaction = self.game.turn_count
			other_ant.prev_interaction = self.game.turn_count
			print('{} + {} -> {}'.format(self.id, other_ant.id, child.id))

	##
	## @brief      Fights to the death with another ant.
	##
	## @param      self       The object
	## @param      other_ant  The other ant
	##
	## @return     { description_of_the_return_value }
	##
	def fight(self, other_ant):
		if self.agent.wins_fight(other_ant.agent.genes):
			#VICTORY IS SWEET
			self.game.ant_list.remove(other_ant)
			self.game.all_sprites_list.remove(other_ant)
			print('{} + {} X {}'.format(self.id, other_ant.id, other_ant.id))
		else:
			#RIP :(
			self.game.ant_list.remove(self)
			self.game.all_sprites_list.remove(self)
			print('{} + {} X {}'.format(self.id, other_ant.id, self.id))

	#establish the list of potential interactions between 2 ants
	interactions = [
		attempt_mate,	#1
		fight 			#2
		]


class Agent:
	"""
	@brief      Class for an agent. Each ant has an agent that tells it what to
	            do. The agents have 'genes' which determine what kind of actions
	            they take.
	"""
	movement_weights = np.array([1,1,2,2])	#see act()
	# Right now, assuming perfectly random genes, ~1/2 of all interactions are a mating event
	interaction_pow_weights = np.array([1,1.2]) # see select_interaction

	##
	## @brief      Constructs the agent. It is possible to have the same agent controlling multiple ants.
	##
	## @param      self   The object
	## @param      genes  The genes of the agent
	##
	def __init__(self, genes=None):
		if genes is None:
			self.genes = (np.random.rand(4,4) - .5)*2
		else:
			self.genes = genes

		#pre-allocating veriables for efficiency reasons. 
		self.movement_inputs = np.empty((1,4))
		self.weighted_movement_inputs = np.empty((1,4))
		self.movement_outputs = np.empty((1,2))
		#eventually I'll use this when I add more ant interactions than just the two.
		self.interaction_range_sizes = np.empty((1,len(Ant.interactions)))

	##
	## @brief      Select which possible ant-ant interaction will occur based on genes
	##
	## @param      self   The object
	## @param      genes  The genes of the other agent
	##
	## @return     index of the relevant action in Ant.interactions
	##
	def select_interaction(self, genes):
		#decide likelihood of each type of interaction based on genes.
		#ants are more likely to fight if they're related. This will hopefully encourage
		#more genetic diversity and also discourage homogenous colonies

		# .5 <= similarity <= 1.5
		# similarity uses 1st order norm to reduce complexity
		similarity = 1.5-(np.linalg.norm((self.genes - genes).flatten(), ord=1) / (2*self.genes.size))

		#the more similar your genes are, the more likely it is that you will fight--raise similarity to the
		#power of the weight
		#TODO: incorporate genetics into this also. Basically make genes determine how aggressive you are.
		self.interaction_range_sizes = np.power(similarity, Agent.interaction_pow_weights)
		#make all the sizes sum to 1
		self.interaction_range_sizes = self.interaction_range_sizes / np.linalg.norm(self.interaction_range_sizes, ord=1)

		#pick a number between 0 and 1, then figure out in what range that number lies. Ranges are scaled above to determine
		#the likelihood that the random number lies within any given range.
		val = random.random()
		running_sum = 0
		for idx, range_size in enumerate(self.interaction_range_sizes):
			running_sum += range_size
			if val <= running_sum:
				return idx

	def act(self, ant):
		"""
		@brief      Causes an ant to take an action (update its state and move
		            on the screen) given its current situation.
		
		@param      self             The object
		@param      ant              The ant that we're causing to act
		@param      ant_collisions   The ants we've collided with
		@param      wall_collisions  The walls we've collided with
		
		@return     none
		"""
		'''
		movement phase inputs:
		[gridx, gridy]
		outputs: []

		action phase inputs:
		[gridx, gridy, collideant, collidewall]
		'''

		init_x = ant.rect.x
		init_y = ant.rect.y
		#use current position and velocity, plus genetic preference, to determine movement.
		self.movement_inputs = np.array([ant.rect.x/SCREEN_WIDTH, ant.rect.y/SCREEN_HEIGHT, ant.prev_dx, ant.prev_dy])
		np.multiply(self.movement_inputs, Agent.movement_weights, out=self.weighted_movement_inputs)
		#generate movement outputs between 0 and 1
		self.movement_outputs = np.matmul(self.movement_inputs, self.genes[0:4,0:2]) / Agent.movement_weights.sum() + .5

		#pick a number between 0 and 1. If it is above the normalized movement_output, then we set dx to 1.
		#if it is below movement_output, we set to 0. This allows genes to set a preference on movement,
		#but not to entirely determine it, or else ants get stuck.
		dx = STEP_SIZE if random.random() > self.movement_outputs[0] else (-1*STEP_SIZE)
		dy = STEP_SIZE if random.random() > self.movement_outputs[1] else (-1*STEP_SIZE)

		#list of ants that we collide with this turn
		ant_collisions = []

		#complete x movement action
		ant.rect.x += dx
		if ant.rect.left < 0 or ant.rect.right > SCREEN_WIDTH:
			#undo if that took us off the screen
			ant.rect.x -= dx
		else:
			#collisions with ants
			ant_collisions.extend(pygame.sprite.spritecollide(ant, ant.game.ant_list, False))
			#collision with walls
			wall_collisions = pygame.sprite.spritecollide(ant, ant.game.wall_list, False)

			#prevent us from moving into an obstacle
			for wall in wall_collisions:
				if dx > 0:
					#moving right, so align our right edge with the left edge of the wall
					ant.rect.right = wall.rect.left
				else:
					#moving left, so do the opposite
					ant.rect.left = wall.rect.right

		#complete y movement action
		ant.rect.y += dy
		if ant.rect.top < 0 or ant.rect.bottom > SCREEN_HEIGHT:
			#undo that if it's taking us off the screen
			ant.rect.y -= dy
		else:
			#collisions with ants
			ant_collisions.extend(pygame.sprite.spritecollide(ant, ant.game.ant_list, False))
			#collision with walls
			wall_collisions = pygame.sprite.spritecollide(ant, ant.game.wall_list, False)

			#prevent us from moving into an obstacle
			for wall in wall_collisions:
				if dy > 0:
					ant.rect.bottom = wall.rect.top
				else:
					ant.rect.top = wall.rect.bottom

		#resolve ant collisions
		for other_ant in ant_collisions:
			#spritecollide is dumb and collides you with yourself
			if ant is not other_ant:
				ant.interact(other_ant)

		#save this iteration's movement results
		ant.prev_dx = ant.rect.x - init_x
		ant.prev_dy = ant.rect.y - init_y

	def mate(self, genes):
		"""
		@brief      Produces offspring genes given this agent's genes and another set of genes. For now, offspring
					occurs no matter what.
		
		@param      genes  The genes to mate with
		
		@return     genes of the new offspring.
		"""

		offspring_genes = np.copy(self.genes)
		#generate offspring's genes through random combination of my genes and their genes
		with np.nditer([offspring_genes, genes], op_flags=[['readwrite'], ['readonly']]) as it:
			for mine, theirs in it:
				mine[...] = mine if (random.randrange(2) == 0) else theirs

		return offspring_genes

	##
	## @brief      Determine the outcome of a fight with another ant
	##
	## @param      self   The object
	## @param      genes  The genes of the enemy ant
	##
	## @return     True iff this agent wins the fight
	##
	def wins_fight(self, genes):
		#for now it's just random selection. May change later.
		return random.randrange(2)

	def get_color(self):
		"""
		Somewhat arbitrarily assigns this ant a color from its genes.
		"""
		return ((self.genes[0:3,0]/2)+.5)*255


class Wall(pygame.sprite.Sprite):
	"""
	@brief      an obstacle that prevents the ant from passing through it
	"""
	def __init__(self, color, width, height):
		"""
		color -- wall color
		width -- wall width
		height - wall height
		"""

		super().__init__()

		#make the generic square that represents this ant
		self.image = pygame.Surface((width, height))
		self.image.fill(color)

		#initialize the object position
		self.rect = self.image.get_rect()

class Game:
	def __init__(self):
		self.turn_count = 0
		self.ant_count = 0
		self.ant_list = pygame.sprite.Group()
		self.wall_list = pygame.sprite.Group()
		#list of all sprites in the game, including foods, ants, and obstacles
		self.all_sprites_list = pygame.sprite.Group()

def main():
	"""
	@brief      main function called from program entry point
	
	@return     None
	"""
	pygame.init()
	screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

	#initialize the game object
	game = Game()

	#create some ants
	for i in range(INITIAL_ANTS):
		#create an ant
		ant = Ant(Agent(),
					#store references to the game object
					game,
					#select a random location for the ant
					random.randrange(SCREEN_WIDTH), random.randrange(SCREEN_HEIGHT))

	#create some obstacles
	for i in range(15):
		#create a randomly sized obstacle
		wall = Wall(BLACK, random.betavariate(2,20)*SCREEN_WIDTH, random.betavariate(2,20)*SCREEN_HEIGHT)
		#select a random location for the ant
		wall.rect.x = random.randrange(SCREEN_WIDTH)
		wall.rect.y = random.randrange(SCREEN_HEIGHT)
		#add the wall to the wall and objects lists
		game.wall_list.add(wall)
		game.all_sprites_list.add(wall)

	#loop until user hits the close button
	done = False
	clock = pygame.time.Clock()

	# ----------------------- MAIN PROGRAM LOOP ---------------------
	while not done:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True

		#clear the screen
		screen.fill(WHITE)

		#ants take their move actions
		game.ant_list.update()

		#draw all the sprites
		game.all_sprites_list.draw(screen)

		#limit the FPS of the game
		clock.tick(FPS)
		game.turn_count += 1

		#update the display with what we've drawn
		pygame.display.flip()

	pygame.quit()


if __name__ == '__main__':
	"""
	Program entry point
	"""
	main()
