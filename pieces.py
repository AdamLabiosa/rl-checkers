#---------------------------------------------
# Checkers Piece Classes for Checkers AI
# Created By: Adam Labiosa
# Inspration From: Jonathan Zia
# Last Edited: 2021
# University of Wisconsin - Madison
# ----------------------------------------------------
import tensorflow as tf
import numpy as np
import pieces as p
import random as r
import state as s
import time as t
import copy as c
import math
import os

#---------------------------------------------
# Normal Piece Class
#---------------------------------------------
class Piece():

	"""Defining attributes of a piece"""

	def __init__(self, color, start_file, start_rank):

		"""Defining initial attributes of piece"""

		# Piece Attributes
		self.name = 'Piece'	# Name
		self.symbol = 'P'		# Symbol for algebraic notation
		self.value = 1			# Value (1 for normal piece)
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		self.start_file = start_file
		self.start_rank = start_rank
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current position
		self.file = start_file
		self.rank = start_rank


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""Determining possible actions for piece"""

		# Requires: piece_list
		# Returns: numpy array

		# The piece can move diagonally forward by one square, or two if capturing.

		# For each tile along one of the four movement vectors, append coordinate if:
		# (1) The index is in bounds
		# (2) There is no piece of the same color
		# (3) There was no piece of the opposite color in the preceding step

		# Initialize action vector:
		# [Forward left, Forward right, Jump Left, Jump Right, 4 zeros]
		if self.name=='Piece':

			action_space = np.zeros((1,8))

			# Initialize coordinate aray
			coordinates = []

			if self.is_active:

				# color = white
				if self.color == 'white':

					# if in range and no piece left
					pos_left_move = False
					if self.file-1 > 0 and self.rank+1 < 9: # in range
						pos_left_move = True
						for piece in piece_list:
							if piece.is_active and piece.file==self.file-1 and piece.rank==self.rank+1:
								# is a piece in the way
								pos_left_move = False

					# if in range and no piece right add
					pos_right_move = False
					if self.file+1 < 9 and self.rank+1 < 9: # in range
						pos_right_move = True
						for piece in piece_list:
							if piece.is_active and piece.file==self.file+1 and piece.rank==self.rank+1:
								# is a piece in the way
								pos_right_move = False

					# if piece left and jump possible add
					pos_left_jump = False
					if self.file-2 > 0 and self.rank+2 < 9: # in range
						for piece in piece_list:
							if piece.is_active and piece.file==self.file-1 and piece.rank==self.rank+1 and piece.color!=self.color:
								# there is a piece to jump
								pos_left_jump = True
						for piece in piece_list:
							if piece.is_active and piece.file==self.file-2 and piece.rank==self.rank+2:
								# there is a piece to jump
								pos_left_jump = False


					# if piece right and jump possible add
					pos_right_jump = False
					if self.file+2 < 9 and self.rank+2 < 9: # in range
						for piece in piece_list:
							if piece.is_active and piece.file==self.file+1 and piece.rank==self.rank+1 and piece.color!=self.color:
								# there is a piece to jump
								pos_right_jump = True
						for piece in piece_list:
							if piece.is_active and piece.file==self.file+2 and piece.rank==self.rank+2:
								# there is a piece to jump
								pos_right_jump = False

				# color = black
				if self.color == 'black':

					# if in range and no piece left
					pos_left_move = False
					if self.file-1 > 0 and self.rank-1 > 0: # in range
						pos_left_move = True
						for piece in piece_list:
							if piece.is_active and piece.file==self.file-1 and piece.rank==self.rank-1:
								# is a piece in the way
								pos_left_move = False

					# if in range and no piece right add
					pos_right_move = False
					if self.file+1 < 9 and self.rank-1 > 0: # in range
						pos_right_move = True
						for piece in piece_list:
							if piece.is_active and piece.file==self.file+1 and piece.rank==self.rank-1:
								# is a piece in the way
								pos_right_move = False

					# if piece left and jump possible add
					pos_left_jump = False
					if self.file-2 > 0 and self.rank-2 > 0: # in range
						for piece in piece_list:
							if piece.is_active and piece.file==self.file-1 and piece.rank==self.rank-1 and piece.color!=self.color:
								# there is a piece to jump
								pos_left_jump = True
						for piece in piece_list:
							if piece.is_active and piece.file==self.file-2 and piece.rank==self.rank-2:
								# there is a piece to jump
								pos_left_jump = False


					# if piece right and jump possible add
					pos_right_jump = False
					if self.file+2 < 9 and self.rank-2 > 0: # in range
						for piece in piece_list:
							if piece.is_active and piece.file==self.file+1 and piece.rank==self.rank-1 and piece.color!=self.color:
								# there is a piece to jump
								pos_right_jump = True
						for piece in piece_list:
							if piece.is_active and piece.file==self.file+2 and piece.rank==self.rank-2:
								# there is a piece to jump
								pos_right_jump = False


				if pos_left_move:
					action_space[0, 0] = 1
					coordinates.append([self.file-1, self.rank+1])
				if pos_right_move:
					action_space[0, 1] = 1
					coordinates.append([self.file+1, self.rank+1])
				if pos_left_jump:
					action_space[0, 2] = 1
					coordinates.append([self.file-2, self.rank+2])
				if pos_right_jump:
					action_space[0, 3] = 1
					coordinates.append([self.file+2, self.rank+2])

				# Convert coordinates to numpy array
				coordinates = np.asarray(coordinates)

			# Return possible moves
			if return_coordinates:
				return coordinates
			else:
				return action_space

		# IS KING
		else:

			action_space = np.zeros((1,8))

			# Initialize coordinate aray
			coordinates = []

			if self.is_active:

				# UP

				# if in range and no piece left
				pos_left_move_up = False
				if self.file-1 > 0 and self.rank+1 < 9: # in range
					pos_left_move_up = True
					for piece in piece_list:
						if piece.is_active and piece.file==self.file-1 and piece.rank==self.rank+1:
							# is a piece in the way
							pos_left_move_up = False

				# if in range and no piece right add
				pos_right_move_up = False
				if self.file+1 < 9 and self.rank+1 < 9: # in range
					pos_right_move_up = True
					for piece in piece_list:
						if piece.is_active and piece.file==self.file+1 and piece.rank==self.rank+1:
							# is a piece in the way
							pos_right_move_up = False

				# if piece left and jump possible add
				pos_left_jump_up = False
				if self.file-2 > 0 and self.rank+2 < 9: # in range
					for piece in piece_list:
						if piece.is_active and piece.file==self.file-1 and piece.rank==self.rank+1 and piece.color!=self.color:
							# there is a piece to jump
							pos_left_jump_up = True
					for piece in piece_list:
						if piece.is_active and piece.file==self.file-2 and piece.rank==self.rank+2:
							# there is a piece to jump
							pos_left_jump_up = False


				# if piece right and jump possible add
				pos_right_jump_up = False
				if self.file+2 < 9 and self.rank+2 < 9: # in range
					for piece in piece_list:
						if piece.is_active and piece.file==self.file+1 and piece.rank==self.rank+1 and piece.color!=self.color:
							# there is a piece to jump
							pos_right_jump_up = True
					for piece in piece_list:
						if piece.is_active and piece.file==self.file+2 and piece.rank==self.rank+2:
							# there is a piece to jump
							pos_right_jump_up = False

				# DOWN

				# if in range and no piece left
				pos_left_move_down = False
				if self.file-1 > 0 and self.rank-1 > 0: # in range
					pos_left_move_down = True
					for piece in piece_list:
						if piece.is_active and piece.file==self.file-1 and piece.rank==self.rank-1:
							# is a piece in the way
							pos_left_move_down = False

				# if in range and no piece right add
				pos_right_move_down = False
				if self.file+1 < 9 and self.rank-1 > 0: # in range
					pos_right_move_down = True
					for piece in piece_list:
						if piece.is_active and piece.file==self.file+1 and piece.rank==self.rank-1:
							# is a piece in the way
							pos_right_move_down = False

				# if piece left and jump possible add
				pos_left_jump_down = False
				if self.file-2 > 0 and self.rank-2 > 0: # in range
					for piece in piece_list:
						if piece.is_active and piece.file==self.file-1 and piece.rank==self.rank-1 and piece.color!=self.color:
							# there is a piece to jump
							pos_left_jump_down = True
					for piece in piece_list:
						if piece.is_active and piece.file==self.file-2 and piece.rank==self.rank-2:
							# there is a piece to jump
							pos_left_jump_down = False


				# if piece right and jump possible add
				pos_right_jump_down = False
				if self.file+2 < 9 and self.rank-2 > 0: # in range
					for piece in piece_list:
						if piece.is_active and piece.file==self.file+1 and piece.rank==self.rank+1 and piece.color!=self.color:
							# there is a piece to jump
							pos_right_jump_down = True
					for piece in piece_list:
						if piece.is_active and piece.file==self.file+2 and piece.rank==self.rank-2:
							# there is a piece to jump
							pos_right_jump_down = False


				if pos_left_move_up:
					action_space[0, 0] = 1
					coordinates.append([self.file-1, self.rank+1])
				if pos_right_move_up:
					action_space[0, 1] = 1
					coordinates.append([self.file+1, self.rank+1])
				if pos_left_jump_up:
					action_space[0, 2] = 1
					coordinates.append([self.file-2, self.rank+2])
				if pos_right_jump_up:
					action_space[0, 3] = 1
					coordinates.append([self.file+2, self.rank+2])
				if pos_left_move_down:
					action_space[0, 4] = 1
					coordinates.append([self.file-1, self.rank-1])
				if pos_right_move_down:
					action_space[0, 5] = 1
					coordinates.append([self.file+1, self.rank-1])
				if pos_left_jump_down:
					action_space[0, 6] = 1
					coordinates.append([self.file-2, self.rank-2])
				if pos_right_jump_down:
					action_space[0, 7] = 1
					coordinates.append([self.file+2, self.rank-2])

				# Convert coordinates to numpy array
				coordinates = np.asarray(coordinates)

			# Return possible moves
			if return_coordinates:
				return coordinates
			else:
				return action_space






	def move(self, action, piece_list, print_move=False, algebraic=True):

		"""Moving piece's position"""

		# Requires:	(1) action (element of action vector), (2) piece list, (3) print move? (4) algebraic notation?
		# Returns:	void

		# Action vector:
		# [Forward left, Forward right, Jump Left, Jump Right, 52 zeros]

		# Temporarily save old position for the purposes of algebraic notation
		old_rank = self.rank
		old_file = self.file
########## WHTIE VS BLACK ##############
		
		if self.name == 'Piece':

			# Is white
			if self.color == 'white':
				# left move
				if action==0:
					self.file = self.file-1
					self.rank = self.rank+1
				# right move
				elif action==1:
					self.file = self.file+1
					self.rank = self.rank+1
				# left jump
				elif action==2:
					self.file = self.file-2
					self.rank = self.rank+2
				# right jump
				elif action==3:
					self.file = self.file+2
					self.rank = self.rank+2

				# Update move counter
				self.move_count += 1

				# If a jump move, remove piece

				piece_remove = False
				if action==2:
					piece_remove = True
					for piece in piece_list:
						if piece.is_active and piece.file==old_file-1 and piece.rank==old_rank+1:
							piece.remove()
							piece_remove = True
							remove_name = piece.name
							break

				if action==3:
					piece_remove = True
					for piece in piece_list:
						if piece.is_active and piece.file==old_file+1 and piece.rank==old_rank+1:
							piece.remove()
							piece_remove = True
							remove_name = piece.name
							break

			# Is black
			else:
				# left move
				if action==0:
					self.file = self.file-1
					self.rank = self.rank-1
				# right move
				elif action==1:
					self.file = self.file+1
					self.rank = self.rank-1
				# left jump
				elif action==2:
					self.file = self.file-2
					self.rank = self.rank-2
				# right jump
				else:
					self.file = self.file+2
					self.rank = self.rank-2

				# Update move counter
				self.move_count += 1

				# If a jump move, remove piece

				piece_remove = False
				if action==2:
					piece_remove = True
					for piece in piece_list:
						if piece.is_active and piece.file==old_file-1 and piece.rank==old_rank-1:
							piece.remove()
							piece_remove = True
							remove_name = piece.name
							break

				if action==3:
					piece_remove = True
					for piece in piece_list:
						if piece.is_active and piece.file==old_file+1 and piece.rank==old_rank-1:
							piece.remove()
							piece_remove = True
							remove_name = piece.name
							break

				# Check for promotion
				if self.color == 'white' and self.rank == 8:
					self.name = 'King'
					self.symbol = 'K'
					self.value = 3

				elif self.color == 'black' and self.rank == 1:
					self.name = 'King'
					self.symbol = 'K'
					self.value = 3


		# Is King
		else:

			# left move up
			if action==0:
				self.file = self.file-1
				self.rank = self.rank+1
			# right move up
			elif action==1:
				self.file = self.file+1
				self.rank = self.rank+1
			# left jump up
			elif action==2:
				self.file = self.file-2
				self.rank = self.rank+2
			# right jump up
			elif action==3:
				self.file = self.file+2
				self.rank = self.rank+2
			# left move down
			elif action==4:
				self.file = self.file-1
				self.rank = self.rank-1
			# right move down
			elif action==5:
				self.file = self.file+1
				self.rank = self.rank-1
			# left jump down
			elif action==6:
				self.file = self.file-2
				self.rank = self.rank-2
			# right jump down
			else:
				self.file = self.file+2
				self.rank = self.rank-2

			# Update move counter
			self.move_count += 1

			# If a jump move, remove piece

			piece_remove = False
			if action==2:
				piece_remove = True
				for piece in piece_list:
					if piece.is_active and piece.file==old_file-1 and piece.rank==old_rank+1:
						piece.remove()
						piece_remove = True
						remove_name = piece.name
						break

			if action==3:
				piece_remove = True
				for piece in piece_list:
					if piece.is_active and piece.file==old_file+1 and piece.rank==old_rank+1:
						piece.remove()
						piece_remove = True
						remove_name = piece.name
						break

			if action==6:
				piece_remove = True
				for piece in piece_list:
					if piece.is_active and piece.file==old_file-1 and piece.rank==old_rank-1:
						piece.remove()
						piece_remove = True
						remove_name = piece.name
						break

			if action==7:
				piece_remove = True
				for piece in piece_list:
					if piece.is_active and piece.file==old_file+1 and piece.rank==old_rank-1:
						piece.remove()
						piece_remove = True
						remove_name = piece.name
						break



		# Print movement if indicated
		file_list = ['a','b','c','d','e','f','g','h']
		if print_move and algebraic:
			if piece_remove:
				print(self.symbol + file_list[old_file-1] + str(old_rank)+ " x " + file_list[self.file-1] + str(self.rank))
			else:
				print(self.symbol + file_list[old_file-1] + str(old_rank) + "-" + file_list[self.file-1] + str(self.rank))
		elif print_move:
			if piece_remove:
				print(self.name + " to " + str(self.file) + "," + str(self.rank) + " taking " + remove_name)
			else:
				print(self.name + " to " + str(self.file) + "," + str(self.rank))


	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False