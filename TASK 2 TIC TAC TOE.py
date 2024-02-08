import random

# Function to print the Tic Tac Toe board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

# Function to check if a player has won
def check_win(board, player):
    for row in board:
        if all([cell == player for cell in row]):
            return True
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False

# Function to check if the board is full
def is_board_full(board):
    return all([cell != " " for row in board for cell in row])

# Minimax algorithm
def minimax(board, depth, is_maximizing):
    if check_win(board, "X"):
        return -1
    if check_win(board, "O"):
        return 1
    if is_board_full(board):
        return 0
    
    if is_maximizing:
        max_eval = -float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    eval = minimax(board, depth + 1, False)
                    board[i][j] = " "
                    max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    eval = minimax(board, depth + 1, True)
                    board[i][j] = " "
                    min_eval = min(min_eval, eval)
        return min_eval

# AI's move using Minimax
def ai_move(board):
    best_move = None
    best_eval = -float("inf")
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = "O"
                move_eval = minimax(board, 0, False)
                board[i][j] = " "
                
                if move_eval > best_eval:
                    best_eval = move_eval
                    best_move = (i, j)
                    
    return best_move

# Main game loop
def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    player_turn = True  # True for player X, False for player O
    
    while not (check_win(board, "X") or check_win(board, "O") or is_board_full(board)):
        print_board(board)
        
        if player_turn:
            row, col = map(int, input("Enter row and column (comma-separated) for your move (e.g., 0,2): ").split(","))
            if board[row][col] != " ":
                print("Invalid move. Cell already occupied.")
                continue
            board[row][col] = "X"
        else:
            print("AI's turn...")
            row, col = ai_move(board)
            board[row][col] = "O"
            
        player_turn = not player_turn
        
    print_board(board)
    if check_win(board, "X"):
        print("Player X wins!")
    elif check_win(board, "O"):
        print("Player O wins!")
    else:
        print("It's a draw!")

# Start the game
play_game()
