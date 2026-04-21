"""
Util file containing several functions useful for evaluating LLMs on puzzle positions from the evaluation set
of puzzles I extracted from the Lichess Puzzle Database.

sample_puzzles(filepath, n) - Samples n puzzles from the provided filepath
get_engine() - Returns an instance of the Stockfish engine
check_position_accuracy(response, fen, best_move_uci, best_move_san, engine) - Checks if the response by a LLM/chess engine is the best move in a position
"""

# imports
import random
import chess
import chess.engine

# Main Linux path
# STOCKFISH_PATH = './stockfish_18_compiled'     # THE BIG FISH

# alternate windows path used for other tests
STOCKFISH_PATH = "stockfish-18.exe"

SF18_JUDGE_DEPTH = 25                          # used to check for alternate puzzle solutions
SF18_JUDGE_TIMEOUT = 10.0                       # limit engine searches to 10 seconds

'''
This function returns a sample of n puzzles from the provided filepath in the form:
puzzles[positions[(FEN, best_move_uci, best_move_san)]]

Each (FEN, best_move_uci, best_move_san) tuple corresponds to a single puzzle position.
'''
def sample_puzzles(filepath: str, n: int) -> list[list[tuple[str, str, str]]]:
    all_puzzles = [] # all puzzles in file
    current_puzzle = [] # current puzzle being processed
    current_position = {} # current puzzle position being processed

    # open the file
    with open(filepath, "r") as f:
        # extract puzzles based on file format
        for line in f:
            line = line.strip()
            # start a new puzzle
            if line == "<|puzzle-start|>":
                current_puzzle = []
            # start a new puzzle position
            elif line == "<|position-start|>":
                current_position = {}
            # set current position
            elif line.startswith("FEN:"):
                current_position["fen"] = line[len("FEN:"):].strip()
            # get best move for current position in UCI
            elif line.startswith("Best move (UCI):"):
                current_position["uci"] = line[len("Best move (UCI):"):].strip()
            # get best move for current position in SAN
            elif line.startswith("Best move (SAN):"):
                current_position["san"] = line[len("Best move (SAN):"):].strip()
            # add current position to current puzzle
            elif line == "<|position-end|>":
                current_puzzle.append((
                    current_position["fen"],
                    current_position["uci"],
                    current_position["san"],
                ))
            # add all positions to the current puzzle
            elif line == "<|puzzle-end|>":
                all_puzzles.append(current_puzzle)

    # return a random sample of n puzzles
    return random.sample(all_puzzles, min(n, len(all_puzzles)))


'''
Returns an instance of the Stockfish engine.
Main function is responsible for calling engine.quit()!
'''
def get_engine() -> chess.engine.SimpleEngine:
    return chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


'''
Checks if a given response by an LLM/chess engine is correct for a given puzzle position.
Returns True if the response matches the recorded best move exactly (UCI or SAN) OR
if the response achieves the same mate-in-N evaluation as the best move (using SF18 at depth=20).
'''
def check_position_accuracy(response: str, fen: str, best_move_uci: str, best_move_san: str, engine: chess.engine.SimpleEngine) -> bool:

    # check if response matches recorded answer exactly
    if response == best_move_uci or response == best_move_san:
        return True

    # try to parse the response as a legal move
    board = chess.Board(fen)
    move = None

    # attempt to parse a UCI move from the response
    try:
        candidate = chess.Move.from_uci(response)
        if candidate in board.legal_moves:
            move = candidate
    except ValueError:
        pass

    # if no UCI move was successfully parsed...
    if move is None:
        # attempt to parse a SAN move from the response
        try:
            move = board.parse_san(response)
        except ValueError:
            return False # return False if no valid UCI or SAN move can be parsed

    # if a valid UCI/SAN move was parsed...
    # check if response gives the same mate evaluation as the recorded answer

    # get the mate depth of the best move from the original position
    og_eval = engine.analyse(board, chess.engine.Limit(depth=SF18_JUDGE_DEPTH,time=SF18_JUDGE_TIMEOUT)).get("score")
    # return False if the og_eval is not mate (something has gone terribly wrong here)
    if og_eval is None or not og_eval.relative.is_mate():
        return False # this should never be called!
    
    # get the mate depth (mate in n moves for the side to move)
    mate_in_n = og_eval.relative.mate() 
    if mate_in_n is None:
        return False # this should never be called!

    # make the provided move (response) and evaluate the resulting position
    board.push(move)
    new_eval = engine.analyse(board, chess.engine.Limit(depth=SF18_JUDGE_DEPTH,time=SF18_JUDGE_TIMEOUT)).get("score")

    # if the provided move yields an invalid evaluation (invalid or not mate)
    # invalid eval
    if new_eval is None or not new_eval.relative.is_mate():
        return False # it isn't the best move
    new_mate = new_eval.relative.mate()

    # eval that is not mate
    if new_mate is None:
        return False

    # after the move, the opponent should face mate in (N-1)
    # relative score from opponent's view should be Mate(-(N-1))
    return new_mate == -(mate_in_n - 1)
