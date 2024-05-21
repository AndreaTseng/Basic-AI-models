# In this project, we will develop an AI game player for a game called Teeko
# implement the minmax algorithm to train the model to make better moves

import random
import numpy as np
import copy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.depth = 3
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        
    def succ(self, state):
        successors = []
        drop_phase = True  # Detecting drop phase by counting the pieces

        num_pieces = sum(row.count(self.my_piece) for row in state)
        if num_pieces >= 4:
            drop_phase = False

        if drop_phase:
            # If it's the drop phase, add the current player's piece to empty spots
            for row in range(len(state)):
                for col in range(len(state[0])):
                    if state[row][col] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[row][col] = self.my_piece
                        successors.append(new_state)
        else:
            # If not drop phase, move a piece to an adjacent space
            for row in range(len(state)):
                for col in range(len(state[0])):
                    if state[row][col] == self.my_piece:
                        # Attempt to move to all adjacent spaces
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue  # Skip no-move
                                new_row, new_col = row + dr, col + dc
                                if (0 <= new_row < len(state)) and (0 <= new_col < len(state[0])) and state[new_row][new_col] == ' ':
                                    new_state = copy.deepcopy(state)
                                    new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                                    successors.append(new_state)
        return successors

        

    def heuristic_game_value(self, state):  # check largest number of pieces connected
        # Access the player's and opponent's pieces from the class attributes
        mine = self.my_piece
        oppo = self.opp

        # for horizontal
        max_this = 0
        max_opponent = 0
        this_count = 0
        count_opponent = 0

        # Check for the longest horizontal and vertical lines for both players
        for i in range(5):
            this_count = state[i].count(mine)
            count_opponent = state[i].count(oppo)
            max_this = max(max_this, this_count)
            max_opponent = max(max_opponent, count_opponent)

            this_count = [state[j][i] for j in range(5)].count(mine)
            count_opponent = [state[j][i] for j in range(5)].count(oppo)
            max_this = max(max_this, this_count)
            max_opponent = max(max_opponent, count_opponent)

        # Check for the longest diagonal connection for both players
        for i in range(5):
            for j in range(5):
                if i + 3 < 5 and j + 3 < 5:
                    this_count = sum(1 for k in range(4) if state[i + k][j + k] == mine)
                    count_opponent = sum(1 for k in range(4) if state[i + k][j + k] == oppo)
                    max_this = max(max_this, this_count)
                    max_opponent = max(max_opponent, count_opponent)

                if i + 3 < 5 and j - 3 >= 0:
                    this_count = sum(1 for k in range(4) if state[i + k][j - k] == mine)
                    count_opponent = sum(1 for k in range(4) if state[i + k][j - k] == oppo)
                    max_this = max(max_this, this_count)
                    max_opponent = max(max_opponent, count_opponent)

        # Check for 2x2 box for both players
        for i in range(4):
            for j in range(4):
                this_count = sum(1 for k in range(2) for l in range(2) if state[i + k][j + l] == mine)
                count_opponent = sum(1 for k in range(2) for l in range(2) if state[i + k][j + l] == oppo)
                max_this = max(max_this, this_count)
                max_opponent = max(max_opponent, count_opponent)

        # Calculate and normalize the heuristic value
        heuristic_value = (max_this - max_opponent) / 4.0  # Normalize to the range -1 to 1
        return heuristic_value

    
    def make_move(self, state):
        drop_phase = sum(sum(1 for cell in row if cell != ' ') for row in state) < 8

        if drop_phase:
            # Drop phase: simply find an empty spot to place a piece
            for i, row in enumerate(state):
                for j, cell in enumerate(row):
                    if cell == ' ':
                        # Return move for placing a new piece
                        return [(i, j)]
        else:
            # Movement phase: use Minimax to find the best move
            _, move_state = self.max_value(state, self.depth)
            move = []

            # Determine the changes between the current state and the move state
            for i in range(5):
                for j in range(5):
                    if state[i][j] != move_state[i][j]:
                        if state[i][j] == ' ':
                            move.insert(0, (i, j))  # Destination of the move
                        else:
                            move.append((i, j))  # Source of the move

            return move

        # Fallback if no move found (should not happen)
        return [(0, 0)]


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
       
        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)
        
    
    def max_value(self, state, depth):
        terminal_result = self.game_value(state)
        # Check if the current state is a terminal state
        if terminal_result != 0:
            return terminal_result, state

        # If maximum depth is reached, use the heuristic function
        if depth == 0:
            return self.heuristic_game_value(state), state

        highest_value = float('-Inf')
        current_best_state = None
        possible_states = self.succ(state)

        # Recursively find the maximum value from the possible states
        for child_state in possible_states:
            result, _ = self.min_value(child_state, depth - 1)
            if result > highest_value:
                highest_value = result
                current_best_state = child_state

        # Return the highest value and the corresponding state
        return highest_value, current_best_state if current_best_state else state

    def min_value(self, state, depth):
        game_result = self.game_value(state)
        # Check if the game is already decided at the current state
        if game_result != 0:
            return game_result, state

        # If maximum depth is reached, use the heuristic function
        if depth == 0:
            return self.heuristic_game_value(state), state

        minimal_value = float('Inf')
        best_state = None
        successor_states = self.succ(state)

        # Iterating through each possible successor state to find the minimal evaluation
        for child_state in successor_states:
            potential_value, _ = self.max_value(child_state, depth - 1)
            if potential_value < minimal_value:
                minimal_value = potential_value
                best_state = child_state

        # Return the minimal value and the corresponding state
        return minimal_value, best_state if best_state else state


    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for i in range(2):
                if state[row][i] != ' ' and state[row][i] == state[row + 1][i + 1] == state[row + 2][i + 2] == \
                        state[row + 3][i + 3]:
                    return 1 if state[row][i] == self.my_piece else -1
        # TODO: check / diagonal wins
        for row in range(3, 5):
            for i in range(2):
                if state[row][i] != ' ' and state[row][i] == state[row - 1][i + 1] == state[row - 2][i + 2] == \
                        state[row - 3][i + 3]:
                    return 1 if state[row][i] == self.my_piece else -1
        
        # TODO: check box wins
        for row in range(4):
            for i in range(4):
                if state[row][i] != ' ' and state[row][i] == state[row][i + 1] == state[row + 1][i] == state[row + 1][
                    i + 1]:
                    return 1 if state[row][i] == self.my_piece else -1
        

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
