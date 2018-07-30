import numpy as np
import copy
import h5py

''' Declaration of constants to represent each possible
 item in the grid world representation. '''
WORLD = {
	'EMPTY': -3,
	'OBSTACLE': -2,
	'AGENT': -1,
	'GOAL': 0
}

# Helper functions for obstacle generation and placement

''' Obtain the number of empty spaces in the grid for
# the wavefront potential function '''
def _getNumUnvisitedSpaces(grid):
	return np.count_nonzero(grid == WORLD['EMPTY'])

''' Get list of unvisited points that have neighbors of a given distance '''
def _getNeighborsD(grid, distance):
	boundaryPointsList = []
	for i in range(len(grid)):
		for j in range(len(grid[i])):
			neighbors = _getNeighbors( (i, j), grid)
			# check if it has a neighbor of distance d but is
			# unvisited itself
			if _checkDistance(neighbors, distance, grid) == 1 \
			and grid[i][j] == WORLD['EMPTY']:
				boundaryPointsList.append( (i, j) )
	return boundaryPointsList

''' Check if there is a neighboring point of some distance '''
def _checkDistance(neighbors, distance, grid):
	return np.any(grid[neighbors.T] == distance)

''' Get a list of valid neighbors for any given point in the grid
for the wavefront potential calculation '''
def _getNeighbors(point, grid):
	y = point[0]
	x = point[1]
	listNeighbors = []
	# Get (y, x - 1)
	if x - 1 >= 0 and grid[y, x - 1] != WORLD['OBSTACLE']:
		listNeighbors.append( (y, x - 1) )
	# Get (y, x + 1)
	if x + 1 <= len(grid[0]) - 1 and grid[y, x + 1] != WORLD['OBSTACLE']:
		listNeighbors.append((y, x + 1))
	# Get (y + 1, x)
	if y + 1 <= len(grid) - 1 and grid[y + 1, x] != WORLD['OBSTACLE']:
		listNeighbors.append((y + 1, x))
	# Get (y - 1, x)
	if y - 1 >= 0 and grid[y - 1, x] != WORLD['OBSTACLE']:
		listNeighbors.append((y - 1, x))
	return listNeighbors

''' Determine if a coordinate is out of bounds in a grid '''
def _out_of_bounds(row, col, num_rows, num_cols):
	return 0 <= rows < num_rows and 0 <= col <= num_cols

''' Class to handle a 2D world representation where a set of agents navigate
from starting positions to goal positions amidst a number of randomly
placed obstacles in the world. '''
class GridWorld2D:
	# Initialize world with its x and y dimensions, the
	# total number of obstacles to be created in the world,
	# and the total number of agents in the world
	def __init__(self, world_rows, world_cols, num_obstacles, num_agents, max_obstacle_dim, saved_state = None):
		# World is a 2D array with empty spots, obstacles,
		# a marker of the agents's current positions, and
		# the goal's positions.
		if saved_state is None:
			# Make sure that the desired world is realizable
			SLACK = 2
			assert(SLACK * (2 * num_agents + num_obstacles) < world_rows * world_cols)

			# Initialize agent and goal locations in the world in addition
			# to initializing all cells in the grid representation in the
			# world as empty
			self.num_agents = num_agents
			self.max_obstacle_dim = max_obstacle_dim
			self.agent_locs = np.zeros((self.num_agents, 2)).astype(int)
			self.agent_goal_locs = np.zeros((self.num_agents, 2)).astype(int)
			self.world = np.zeros((world_rows, world_cols)) + WORLD['EMPTY']
			self.agent_distances_to_goal = self._get_agent_distances_to_goal()

			# Randomly set the agent and goal locations
			self._set_random_agent_init_locs()
			self._set_random_goal_locs()

			# Initialize a obstacle count and list of obstacles,
			# and modify these as obstacles are generated and
			# valid ones are added to the world until the total
			# number of obstacles desired have been added.
			obstacle_count = 0
			self.obs_list = []
			while obstacle_count < num_obstacles:
				obstacle_count += self._add_rect_obstacle(self.max_obstacle_dim)
		else:
			# Saved state simply contains all the information in get_world_representation
			self.agent_locs = saved_state['agent_locs']
			self.agent_goal_locs = saved_state['agent_goal_locs']
			self.world = saved_state['world']
			self.num_agents = len(self.agent_locs)
			self.max_obstacle_dim = None
			self.agent_distances_to_goal = self._get_agent_distances_to_goal()
			self.obs_list = zip(*np.where(saved_state == WORLD['OBSTACLE']))

	''' Takes one hot vector input [LEFT, RIGHT, UP, DOWN]
	and indicates whether the move was successful. Used
	to update world after a particular agent takes an action '''
	def take_action(self, action, agent_index):

		# Make sure a valid action is passed in:
		assert(np.sum(action) <= 1)
		# Set new agent position
		agent_pos = self.agent_locs[agent_index]
		moved = False
		try_move = True

		# Move Left
		if action[0] == 1:
			new_position = np.array([agent_pos[0], agent_pos[1] - 1])
		# Move Right
		elif action[1] == 1:
			new_position = np.array([agent_pos[0], agent_pos[1] + 1])
		# Move Up
		elif action[2] == 1:
			new_position = np.array([agent_pos[0] - 1, agent_pos[1]])
		# Move Down
		elif action[3] == 1:
			new_position = np.array([agent_pos[0] + 1, agent_pos[1]])
		else:
			try_move = False

		# If agent tried to move, the new position is not out of bounds,
		# and the new position is not an obstacle/other agent, move the agent
		# there
		if try_move and \
		not _out_of_bounds(new_position[0], new_position[1], len(self.world), len(self.world[0])) and \
		self.world[new_position[0], new_position[1]] != WORLD['OBSTACLE'] and \
		self.world[new_position[0], new_position[1]] != WORLD['AGENT']:
			self.world[agent_pos[0], agent_pos[1]] = WORLD['EMPTY']
			self.world[new_position[0], new_position[1]] = WORLD['AGENT']
			self.agent_locs[agent_index] = new_position
			moved = 1
			# If you make a move update the agent distances to goals as well
			self.agent_distances_to_goal = self._get_agent_distances_to_goal()

		return moved

	''' Generates a message (numpy array) between agents. '''
	def get_agent_message(self, send_agent_index, n, encoder=None):
		if encoder is None:
			return self.get_neighborhood_state(n, send_agent_index)
		else:
			return encoder(self.get_neighborhood_state(n, send_agent_index))

	''' Decodes a received message. '''
	def decode_agent_message(self, msg, decoder=None):
		if decoder is None:
			return msg
		else:
			return decoder(msg)

	''' Returns a NxN neighborhood of a specific agent centered on the robot location
	in the world. Out of bounds cells are treated as obstacles. '''
	def get_neighborhood_state(self, n, agent_index):

		# Get row and column ranges for the neighborhood state grid
		# and initialize the neighborhood grid.
		rows = (self.agent_locs[agent_index][0] - int(n/2), self.agent_locs[agent_index][0] + int(n/2))
		cols = (self.agent_locs[agent_index][1] - int(n/2), self.agent_locs[agent_index][1] + int(n/2))
		neighborhood_grid = np.ones((n, n)) * WORLD['OBSTACLE']

		# Fill in neighborhood grid from spaces in the world around the robot
		# location.
		# for row in range(rows[0], rows[1] + 1):
		# 	for col in range(cols[0], cols[1] + 1):
		# 		if _out_of_bounds(row, col, len(self.world), len(self.world[0])) == False:
		# 			neighborhood_grid[row - rows[0], col - cols[0]] = self.world[row][col]
		# TODO: Check this math
		left = max(0, cols[0])
		right = min(len(self.world[0]), cols[1] + 1)
		top = max(0, rows[0])
		bottom = min(len(self.world), rows[1] + 1)
		offset = top - rows[0], left - cols[0]
		local_grid = self.world[left:right, top:bottom]
		localy, localx = local_grid.shape
		neighborhood_grid[offset[0]:offset[0] + localy, offset[1]:offset[1]+localx]
		return neighborhood_grid

	''' Returns current world representation '''
	def get_world(self):
		return self.world

	''' Determines whether a particular agent has reached its goal '''
	def goal_test(self, agent_index):
		return np.array_equal(self.agent_locs[agent_index], self.agent_goal_locs[agent_index])

	''' Determines whether all agents have reached their goal '''
	def game_over_test(self):
		return np.array_equal(self.agent_locs, self.agent_goal_locs)

	''' Computes neighborhood state and displays it similar
	to the way the world is displayed '''
	def display_neighborhood_state(self, n, agent_index):
		neighborhood_grid = self.get_neighborhood_state(n, agent_index)

		lineStr = ""
		for j in range(len(neighborhood_grid[0]) + 2):
			lineStr += "X "
		print lineStr
		for i in range(len(neighborhood_grid)):
			# Initialize string for that line
			lineStr = "X "
			for j in range(len(neighborhood_grid[i])):
				# Print O for obstacle, G for goal,
				# R for robot, and E for empty
				if neighborhood_grid[i][j] == WORLD['OBSTACLE']:
					lineStr += "O "
				elif neighborhood_grid[i][j] == WORLD['GOAL']:
					lineStr += "G "
				elif neighborhood_grid[i][j] == WORLD['AGENT']:
					lineStr += "A "
				else:
					lineStr += "  "
			print lineStr + "X "
		lineStr = ""
		for j in range(len(neighborhood_grid[0]) + 2):
			lineStr += "X "
		print lineStr
		return

	''' Display the world in a nice 2D format, with an X outside the
	border of the world, an O for obstacles, a G for the goal,
	an R for the robot, and a space character for an empty space
	in the grid. '''
	def display_world(self):
		lineStr = ""
		for j in range(len(self.world[0]) + 2):
			lineStr += "X "
		print lineStr
		for i in range(len(self.world)):
			# Initialize string for that line
			lineStr = "X "
			for j in range(len(self.world[i])):
				# Print O for obstacle, G for goal,
				# R for robot, and E for empty
				if self.world[i][j] == WORLD['OBSTACLE']:
					lineStr += "O "
				elif self.world[i][j] == WORLD['GOAL']:
					lineStr += "G "
				elif self.world[i][j] == WORLD['AGENT']:
					lineStr += "A "
				else:
					lineStr += "  "
			print lineStr + "X "
		lineStr = ""
		for j in range(len(self.world[0]) + 2):
			lineStr += "X "
		print lineStr
		return

	''' Return dictionary of robot location, goal location,
	and obstacle locations in the world. '''
	def get_world_representation(self):
		return {
			'agent_locs': self.agent_locs,
			'agent_goal_locs': self.agent_goal_locs,
			'obs_list': self.obs_list
		}
	''' Allows you to save a world representation so that
	later you simply reload this into the GridWorld2D
	object instead of generating the world again '''
	def save_world(self, h5py_file_obj, dataset_name):
		h5py_file_obj.create_dataset(dataset_name + "_agent_locs", data = self.agent_locs)
		h5py_file_obj.create_dataset(dataset_name + "_agent_goal_locs", data = self.agent_goal_locs)
		h5py_file_obj.create_dataset(dataset_name + "_world", data = self.world)

	''' Returns a random empty coordinate for the top left
	corner of an obstacle. '''
	def _generate_point_obstacle_at_empty_pos(self):
		obs_row = np.random.randint(0, len(self.world))
		obs_col = np.random.randint(0, len(self.world[0]))

		while self.world[obs_row, obs_col] != WORLD['EMPTY']:
			obs_row = np.random.randint(0, len(self.world))
			obs_col = np.random.randint(0, len(self.world[0]))

		return (obs_row, obs_col)

	'''Obtain all the coordinates of a rectangle in the world
	given the top left corner, the number of desired rows,
	and the number of desired columns in the rectangle. '''
	def _getRectangleCoordinates(self, rows, cols, tl_coor):
		rectCoors = []

		# Add all rectangle coordinates to a list to be returned
		# that are in-bounds in the world.
		for i in range(tl_coor[0], tl_coor[0] + rows):
			for j in range(tl_coor[1], tl_coor[1] + cols):
				if not _out_of_bounds(i, j, len(self.world),
				len(self.world[0]) ):
					rectCoors.append((i, j))

		return rectCoors

	''' Returns the coordinates of a randomly generated
	rectangular obstacle that has coordinates solely in
	a region of the world that is currently empty '''
	def _generate_rect_obstacle_at_empty_pos(self, max_obstacle_dim):

		# Generate an empty top-left corner for the obstacle
		tl_coor = self._generate_point_obstacle_at_empty_pos()

		# Get top left coordinates that we know are empty
		obs_tl_row = tl_coor[0]
		obs_tl_col = tl_coor[1]

		# Generate a bottom right corner. Here we enforce that no
		# obstacle has either dimension larger than 5 units in length
		row_last = np.random.randint(obs_tl_row, obs_tl_row + max_obstacle_dim)
		col_last = np.random.randint(obs_tl_col, obs_tl_col + max_obstacle_dim)

		rectangle_coordinates = []

		# Work up from the top left corner and fill in the largest rectangle
		# contained between the generated top left and bottom right corners
		# while ensuring that the region in the world corresponding to these
		# coordinates is empty.
		for i in range(obs_tl_row, row_last):
			for j in range(obs_tl_col, col_last):
				# Get rectangle coordinates
				rectCoor = self._getRectangleCoordinates(i - obs_tl_row + 1,
				j - obs_tl_col + 1, tl_coor)
				all_free = True
				# Check if all of these coordinates are empty in the world
				for k in range(len(rectCoor)):
					if self.world[rectCoor[k][0],
					rectCoor[k][1]] != WORLD['EMPTY']:
						all_free = False

				# If all of these coordinates are free, add them to the list
				# of rectangle coordinates to fill up
				if all_free == True:
					for coor in rectCoor:
						rectangle_coordinates.append(coor)
				# Otherwise, our rectangle is as big as it can be without
				# running into non-empty spaces, so we return the rectangle
				# coordinates
				else:
					return rectangle_coordinates

		# Return list of rectangle coordinates if we have not already
		return rectangle_coordinates

	''' Add a rectangular obstacle to the world '''
	def _add_rect_obstacle(self, max_obstacle_dim):

		obstacle_added = 0

		# Generate obstacle and add it to the world
		coordinates = self._generate_rect_obstacle_at_empty_pos(max_obstacle_dim)
		for coor in coordinates:
			self.world[coor[0], coor[1] ] = WORLD['OBSTACLE']

		# If no obstacle was actually created, indicate this
		if len(coordinates) == 0:
			return obstacle_added

		# Check if the obstacle can be added while ensuring there exists
		# a path between every (agent, agent_goal) pair for every agent
		# its corresponding goal. If so, add it and indicate that it was
		# added. We are specifically ensuring that even with the obstacles
		# in the environment, there exists a path between every agent and
		# its goal assuming agents cannot follow paths that contain obstacles
		agent_distances_to_goal = self._get_agent_distances_to_goal()
		if agent_distances_to_goal is not None:
			obstacle_added = 1
			self.obs_list.append(coordinates)

		# If the obstacle cannot be added without eliminating paths
		# between an agent and its corresponding goal, erase it from
		# the world
		else:
			for coor in coordinates:
				self.world[coor[0], coor[1]] = WORLD['EMPTY']

		if obstacle_added:
			self.agent_distances_to_goal = agent_distances_to_goal

		return obstacle_added

	''' Get the distance of each agent to its corresponding goal,
	assuming that the agent cannot move in a square occupied by
	an obstacle or occupied by another agent. If there is no
	path between an agent and its goal, return None.
	Distances to goals computed assuming agents cannot
	follow paths through obstacles, but location of other agents
	are ignored, even though no two agents can occupy the same
	square. This is because the other agents could presumably
	move and thus their location is not a big factor in the
	true distance between an agent and its goal '''
	def _get_agent_distances_to_goal(self):
		agent_distances_to_goal = np.zeros(self.num_agents)
		for i in range(len(self.agent_locs)):
			agent_distance_to_goal = self._wave_front(i)[self.agent_locs[i][0], self.agent_locs[i][1]]

			# Handle case where agent is on top of goal
			if np.array_equal(self.agent_locs[i], self.agent_goal_locs[i]):
				agent_distance_to_goal = 0

			if agent_distance_to_goal == WORLD['EMPTY']:
				return None
			else:
				agent_distances_to_goal[i] = agent_distance_to_goal

		return agent_distances_to_goal

	''' Get wave_front coloring for the world using the wave front
	potential method. In order to ensure that every agent can
	reach its goal, can simply expand outward from that agents
	goal position and see if we can reach the agent's position.
	To do this, we simply operate in a modified gird representation
	where the agents location is marked as empty as are the goal
	positions for every other agent. Then, if we can expand from
	the agent goal to the agent's position successfully, clearly
	the agent can reach its goal. Also tells you distance of
	each agent from their goal.'''
	def _wave_front(self, agent_index):
		# Make copy of the world to fill in
		grid = copy.deepcopy(self.world)
		# Get agent position
		agent_pos = self.agent_locs[agent_index]

		# Treat the other goals as empty cells for wavefront computation
		for i in range(len(self.agent_goal_locs)):
			if i != agent_index:
				grid[self.agent_goal_locs[i][0], self.agent_goal_locs[i][1]] = WORLD['EMPTY']

		# Treat the agent position as an empty cell that can be accessed
		# by the wave-front potential method
		grid[agent_pos[0], agent_pos[1]] = WORLD['EMPTY']

		# Initialize to 0 so we first check distance from goal position
		d = 0
		# Iterate until all spaces have been visited
		while _getNumUnvisitedSpaces(grid) > 0:
			# Get current points to update
			current_points = _getNeighborsD(grid, d)
			# Update distance until the agent position is reached
			# and then return the grid

			# If there are no points left to visit, terminate the algorithm
			if not current_points:
				return grid
			# If the agent position is on the horizon, set the distance
			# from the agent to the goal in the wavefront grid and terminate
			# the algorithm
			elif any((agent_pos == x).all() for x in current_points):
				grid[agent_pos[0], agent_pos[1]] = d + 1
				return grid
			# Otherwise update the distances
			else:
				for i in range(len(current_points)):
					grid[current_points[i][0], current_points[i][1]] = d + 1

			# Go to next layer of distances
			d += 1

		return grid

	''' Set random initial agent locations in empty spaces in the world '''
	def _set_random_agent_init_locs(self):
		for i in range(self.num_agents):
			row = np.random.randint(0, len(self.world) )
			col = np.random.randint(0, len(self.world[0]))

			while self.world[row, col] != WORLD['EMPTY']:
				row = np.random.randint(0, len(self.world) )
				col = np.random.randint(0, len(self.world[0]))

			self.world[row, col] = WORLD['AGENT']
			self.agent_locs[i] = np.array([row, col])

	''' Set random agent goal locations in empty spaces in the world '''
	def _set_random_goal_locs(self):
		for i in range(self.num_agents):
			row = np.random.randint(0, len(self.world) )
			col = np.random.randint(0, len(self.world[0]))

			while self.world[row, col] != WORLD['EMPTY']:
				row = np.random.randint(0, len(self.world) )
				col = np.random.randint(0, len(self.world[0]))

			self.world[row, col] = WORLD['GOAL']
			self.agent_goal_locs[i] = np.array([row, col])

# Test basic functionality
if __name__ == "__main__":
	g_world = GridWorld2D(7, 7, 10, 2, 5)
	print "Initial World"
	g_world.display_world()
	print ""
	moved = g_world.take_action([1, 0, 0, 0], 0)
	if moved:
		print "AGENT MOVED"
	else:
		print "AGENT DIDNT MOVE"
	print ""
	print "World after Action"
	g_world.display_world()
	print ""
	print "Local State of Agent"
	g_world.display_neighborhood_state(5, 0)
	print "AGENT DISTANCES"
	print g_world.agent_distances_to_goal
	print ""
	print "World after Reloading from HDF5"
	h5f = h5py.File('./data/grid_world_data.h5', 'w')
	g_world.save_world(h5f, 'GridWorld2D_1')
	h5f.close()

	h5f = h5py.File('data/grid_world_data.h5','r')
	loaded = {}
	loaded['agent_locs'] = h5f['GridWorld2D_1_agent_locs'][:]
	loaded['agent_goal_locs'] =  h5f['GridWorld2D_1_agent_goal_locs'][:]
	loaded['world'] = h5f['GridWorld2D_1_world'][:]
	h5f.close()
	loaded_g_world = GridWorld2D(None, None, None, None, None, loaded)
	loaded_g_world.display_world()

	print "AGENT DISTANCES"
	print loaded_g_world.agent_distances_to_goal
