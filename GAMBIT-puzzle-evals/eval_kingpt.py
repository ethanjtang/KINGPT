"""
This program tests different KINGPT checkpoints on a sample of mate-in-X puzzles.
It is identical to eval_all_models.py in my GAMBIT repo (https://github.com/ethanjtang/GAMBIT) except for KINGPT inference.
KINGPT was trained using a fork of karpathy's nanoGPT repository (https://github.com/karpathy/nanoGPT).

A move is treated as correct if it matches the best move recorded OR
yields a mate score equivalent to the best move (decreases the mate-in-X evaluation by 1).
ex) Mate in 2 -> Mate in 1

Please refer to puzzle_utils.py for documentation on the functions:
sample_puzzles() - handles puzzle sampling 
get_engine() - creates custom instances of Stockfish
check_position_accuracy() - checks if the response by a LLM/chess engine is the best move in a position
                            (including alternative solutions!)
"""

from __future__ import annotations # for some strange issue with function definitions

# imports
import os
import sys
import chess
import chess.engine
import torch

import pickle # KINGPT vocab file
from contextlib import nullcontext # for torch device setup
from model import GPTConfig, GPT  # KINGPT tokenizer

from puzzle_utils import sample_puzzles, get_engine, check_position_accuracy

# constants
PUZZLES_DIR = 'puzzles' # for full puzzle sets
SAMPLE_DIR = 'samples'  # save sample of N puzzles for reuse/testing different models
N_PUZZLES = 100         # number of puzzles to test on for each theme
SF18_DEPTH = 20         # depth for Stockfish variants (besides the ground truth)
SF18_TIMEOUT = 10.0     # 10s timeout for Stockfish variants 

META_PATH = 'kingpt-models/meta.pkl'  # shared vocab file for all KINGPT checkpoints

# ============================
# CONFIG
# ============================

# dict to associate puzzle theme with (filepath, mate_depth) pairs
# filepath for filepath (duh)
# UNUSED: mate_depth as an arg to build "cheating" prompts
PUZZLE_FILES = {
    'mateIn1': (os.path.join(PUZZLES_DIR, 'validation_puzzles_mateIn1.txt'), 1),
    'mateIn2': (os.path.join(PUZZLES_DIR, 'validation_puzzles_mateIn2.txt'), 2),
    'mateIn3': (os.path.join(PUZZLES_DIR, 'validation_puzzles_mateIn3.txt'), 3),
}

# List of KINGPT checkpoints (trained by yours truly)
# Maps display name -> path to checkpoint file (.pt)
# All checkpoints share the same META_PATH as their vocab file
KINGPT_MODELS = {
    'kingpt-woodpecker': 'kingpt-models/kingpt_woodpecker_1m-iters.pt',
    'kingpt-beaver': 'kingpt-models/kingpt_beaver_50k-iters.pt',
    'kingpt-beaver-OVERFIT': 'kingpt-models/kingpt_beaver_500k-iters_massivelyoverfit.pt', # testing a really overfitted version for fun
    'kingpt-chimera': 'kingpt-models/kingpt_chimera_1m-iters.pt',
    # Note that the selfplay model was trained only for 50k iters due to val loss converging much quicker
    # due to 1k selfplay games giving a lot less unique positions than 12m puzzles
}

# ============================
# PUZZLE STUFF
# ============================

'''
Helper function used to save sample of N puzzles to a separate .txt file.

filepath - filepath to save to
puzzles - list of puzzles
'''
def save_sample_puzzles(filepath: str, puzzles: list) -> None:
    # Write each puzzle to output file (following the same format as input puzzle files)
    with open(filepath, 'w', encoding='utf-8') as f:
        for puzzle in puzzles:
            f.write('<|puzzle-start|>\n')
            for (fen, uci, san) in puzzle:
                f.write('<|position-start|>\n')
                f.write(f'FEN: {fen}\n')
                f.write(f'Best move (UCI): {uci}\n')
                f.write(f'Best move (SAN): {san}\n')
                f.write('<|position-end|>\n')
            f.write('<|puzzle-end|>\n')

'''
Helper function used to sample N puzzles from each theme.

returns a dict, {puzzles_by_theme} containing keys:
puzzles - sample list of puzzles
mate_depth - info arg used for "cheating" prompts
'''
def fetch_puzzle_sample() -> dict:

    # make sample puzzles dir
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    puzzles_by_theme = {}

    # For each puzzle theme...
    for theme, (source_filepath, mate_depth) in PUZZLE_FILES.items():

        # Either load an existing sample if it exists or write to a new file if no sample exists
        sample_filepath = os.path.join(SAMPLE_DIR, f'{theme}_sample.txt')

        # sample exists, load from existing
        if os.path.exists(sample_filepath):
            print(f'  {theme}: loading existing sample from {sample_filepath}')
            puzzles = sample_puzzles(sample_filepath, N_PUZZLES)
        # sample doesn't exist, sample and write to new file
        else:
            print(f'  {theme}: sampling {N_PUZZLES} puzzle(s) from source and saving to {sample_filepath}')
            # sample N puzzles for theme
            puzzles = sample_puzzles(source_filepath, N_PUZZLES)
            # write new sample to file
            save_sample_puzzles(sample_filepath, puzzles)
        puzzles_by_theme[theme] = (puzzles, mate_depth)

    # (puzzles, mate_depth)
    return puzzles_by_theme

'''
Helper function which attempts to parse a valid UCI or SAN format move from LLM output.
Takes the first, leftmost match, with UCI format matches having priority over SAN matches.

Takes as input:
text - LLM output
board - current chess board for LLM-generated move
Returns None if no legal move is able to be parsed from the current board state.
'''
def parse_move(text: str, board: chess.Board) -> str | None:

    # get all words from LLM response
    words = text.split()

    # parse all possible substrings which could contain valid chess moves
    substrings = [
        ' '.join(words[i:j])
        for i in range(len(words))
        for j in range(i + 1, len(words) + 1)
    ]

    # For each candidate move in the list of substrings...
    for candidate in substrings:
        # I will claim the Royal Sceptre and ...
        
        # Attempt to parse a UCI move
        try:
            move = chess.Move.from_uci(candidate.lower())
            if move in board.legal_moves:
                return candidate.lower()
        except ValueError:
            pass
    
    # If no valid UCI move was parsed..
    # For each candidate move in the list of substrings...
    for candidate in substrings:
        try:
            # Attempt to parse a SAN move
            move = board.parse_san(candidate)
            # If one is parsed, convert it to UCI
            # SAN is a really horrible format for everyone involved (even humans!)
            if move in board.legal_moves:
                return move.uci()
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            pass

    return None

# ============================
# THE BIG FISH
# ============================

'''
Helper function to return a configured version of Stockfish 18.

Returns the configured engine as a chess.engine.SimpleEngine object.
'''
def get_sf_player_engine(skill: int | None) -> chess.engine.SimpleEngine:
    engine = get_engine()
    if skill is not None:
        engine.configure({'Skill Level': skill})
    return engine

'''
Helper function to return the best move in a given position.
Using Stockfish 18 at depth=20.

Returns the best move in UCI format as a string.
'''
def get_ground_truth_move(engine: chess.engine.SimpleEngine, fen: str) -> str:
    board = chess.Board(fen)
    result = engine.play(board, chess.engine.Limit(depth=25))
    return result.move.uci() if result.move else ''

'''
Helper function to return the best move in a given position.
Using the provided engine at N depth.
engine - chess engine to use
fen - chess position
depth - chess engine depth to search

Returns the best move (according to the engine) in UCI format as a string.
'''
def get_engine_move(engine: chess.engine.SimpleEngine, fen: str, depth: int) -> str:
    board = chess.Board(fen)
    result = engine.play(board, chess.engine.Limit(depth=depth,time=SF18_TIMEOUT))
    return result.move.uci() if result.move else ''

'''
Helper function to evaluate a SF18 model variant on a sample of puzzles from X theme.
model_name - SF18 variant name
depth - depth used by variant
puzzles_by_theme - sample of N puzzles from X themes
judge_engine - ground truth engine, used to check for alternative solutions for mate-in-X puzzles
               (since multiple moves can lead to mate)
'''
def evaluate_sf_model(model_name: str, skill: int | None, depth: int, puzzles_by_theme: dict, judge_engine: chess.engine.SimpleEngine):
    
    # SF variant name
    print(f'\n{"=" * 70}')
    print(f'MODEL: {model_name}')
    print(f'{"=" * 70}')

    # initialize engine
    player_engine = get_sf_player_engine(skill)

    # position/puzzle-wide results
    total_positions_correct = 0
    total_positions = 0
    total_puzzles_solved = 0
    total_puzzles = 0

    # For each puzzle theme...
    for theme, (puzzles, _mate_depth) in puzzles_by_theme.items():

        # print out theme being evaluated and number of puzzles in sample
        print(f'\n  Theme: {theme} ({len(puzzles)} puzzles)')
        print(f'  {"-" * 60}')

        # theme-wide results
        theme_positions_correct = 0
        theme_positions_total = 0
        theme_puzzles_solved = 0

        # For each puzzle...
        for puzzle_idx, puzzle in enumerate(puzzles):

            # puzzle-wide results
            puzzle_positions_correct = 0

            print(f'\n  Puzzle {puzzle_idx + 1}/{len(puzzles)}:')

            # For each puzzle position...
            for pos_idx, (fen, best_uci, best_san) in enumerate(puzzle):

                # get the 'best' move from the SF18 variant being evaluated
                response = get_engine_move(player_engine, fen, depth)
                
                # check if the move is actually accurate
                correct = check_position_accuracy(response, fen, best_uci, best_san, judge_engine)
                status = 'CORRECT' if correct else 'WRONG'

                # print out position-specific results
                print(f'    Position {pos_idx + 1}:')
                print(f'      FEN:           {fen}')
                print(f'      Best move UCI: {best_uci}')
                print(f'      Engine played: {response}')
                print(f'      Result:        [{status}]')

                # update results if the position was solved correctly by the variant
                if correct:
                    puzzle_positions_correct += 1
                    theme_positions_correct += 1
                    total_positions_correct += 1
                theme_positions_total += 1
                total_positions += 1

            # check if the variant solved the puzzle correctly (100% accuracy for all puzzle positions)
            puzzle_solved = puzzle_positions_correct == len(puzzle)

            # update results if the puzzle was solved correctly
            if puzzle_solved:
                theme_puzzles_solved += 1
                total_puzzles_solved += 1
            total_puzzles += 1

            # print final puzzle-wide result and position-wide accuracy
            puzzle_status = 'SOLVED' if puzzle_solved else 'FAILED'
            print(f'    Puzzle result: [{puzzle_status}] ({puzzle_positions_correct}/{len(puzzle)} positions correct)')

        # calculate position and puzzle-wide accuracy for all puzzles for the theme being evaluated
        theme_pos_acc = theme_positions_correct / theme_positions_total * 100 if theme_positions_total > 0 else 0
        theme_puzzle_acc = theme_puzzles_solved / len(puzzles) * 100 if puzzles else 0

        # print out theme-wide statistics
        print(f'\n  {theme} summary:')
        print(f'    Puzzles solved:    {theme_puzzles_solved}/{len(puzzles)} ({theme_puzzle_acc:.1f}%)')
        print(f'    Positions correct: {theme_positions_correct}/{theme_positions_total} ({theme_pos_acc:.1f}%)')

    # calculate overall position and puzzle-wide accuracy for all puzzles across all themes
    overall_pos_acc = total_positions_correct / total_positions * 100 if total_positions > 0 else 0
    overall_puzzle_acc = total_puzzles_solved / total_puzzles * 100 if total_puzzles > 0 else 0

    # print overall statistics
    print(f'\n  OVERALL ({model_name}):')
    print(f'    Puzzles solved:    {total_puzzles_solved}/{total_puzzles} ({overall_puzzle_acc:.1f}%)')
    print(f'    Positions correct: {total_positions_correct}/{total_positions} ({overall_pos_acc:.1f}%)')

    # IMPORTANT: close the engine!
    player_engine.quit()

    # return overall results for display
    return (total_puzzles_solved, total_puzzles, total_positions_correct, total_positions, None)

# ============================
# KINGPT EVALUATION
# ============================

'''
Builds a prompt for KINGPT, matching the format used during training.

fen - current puzzle position
'''
def build_prompt_kingpt(fen: str) -> str:
    return (
        '<|position-start|>\n'
        f'FEN: {fen}\n'
        'Best move (UCI): '
    )

'''
This helper function loads a KINGPT checkpoint and its vocab file.
Copy and pasted from the nanoGPT directory's sample.py file.

ckpt_path - Path of KINGPT checkpoint
meta_path - Path of KINGPT vocab file

Returns (model, encode, decode, device)
model - KINGPT instance in eval mode
encode - char -> token mapping
decode - token -> char mapping
device - model to load KINGPT instance on
'''
def load_kingpt(ckpt_path: str, meta_path: str):
    device = 'cuda'

    # load checkpoint and reconstruct model
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # load character-level vocab from meta.pkl
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return model, encode, decode, device

'''
This helper function generates a KINGPT response to a prompt.

model - KINGPT instance in eval mode
encode - char -> token mapping
decode - token -> char mapping
device - model to load KINGPT instance on

Returns model response as a decoded string (only including new tokens past the initial prompt)
'''
def generate_kingpt_response(model, encode, decode, prompt: str, device: str) -> str:

    # set up autocast context
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else nullcontext()

    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        with ctx:
            # we only need 10 tokens since we are expecting a single move in UCI format
            y = model.generate(x, max_new_tokens=10, temperature=1.0, top_k=200)

    # decode only new tokens (not the prompt)
    new_tokens = y[0][len(start_ids):]
    return decode(new_tokens.tolist()).strip()

'''
Evaluates a loaded KINGPT model instance on N samples of X puzzle themes.
Same structure as evaluate_llm_pass, besides there being no cheating variant 
(KINGPT's FEN + best move pair vocab doesn't let me do it without errors)
and it wouldn't help anyways because...
KINGPT is a small/stupid language model, so it literally doesn't understand natural language.
Which kind of goes against the entire point of LLMs as general tools, but KINGPT is still technically an LLM.
I could babble on about this forever but this is likely already somewhere in the paper I wrote.

stockfish my beloved
why date claude/gemini when you have THE BIG FISH

label - label for final results display
model - KINGPT instance
encode - tokenizer for KINGPT
decode - detokenizer for KINGPT (I don't think this is quite accurate but oh well! hello dear reviewer, please be nice my brain is degrading from AI use ouchie owie adsbuoasdfuivyihog)
device - the device KINGPT is loaded on
puzzles_by_theme - sample of N puzzles from X themes
judge_engine - SF18 instance running at depth=25, used to check for alternative solutions to mate-in-X puzzles
'''
def evaluate_kingpt_pass(label: str, model, encode, decode, device: str,
                         puzzles_by_theme: dict, judge_engine: chess.engine.SimpleEngine):

    # print stuff
    print(f'\n{"=" * 70}')
    print(f'MODEL: {label}')
    print(f'{"=" * 70}')

    # results for different levels
    total_positions_correct = 0
    total_positions = 0
    total_puzzles_solved = 0
    total_puzzles = 0

    # measures "sanity" of each LLM
    # number of parses failed = 1 / sanity
    total_parse_failed = 0

    # For each puzzle theme...
    for theme, (puzzles, mate_depth) in puzzles_by_theme.items():

        # print it
        print(f'\n  Theme: {theme} ({len(puzzles)} puzzles)')
        print(f'  {"-" * 60}')

        # theme-wide results
        theme_positions_correct = 0
        theme_positions_total = 0
        theme_puzzles_solved = 0
        theme_parse_failed = 0

        # For each puzzle...
        for puzzle_idx, puzzle in enumerate(puzzles):

            # track num puzzle positions solved
            puzzle_positions_correct = 0
            print(f'\n  Puzzle {puzzle_idx + 1}/{len(puzzles)}:')

            # For each puzzle...
            for pos_idx, (fen, best_uci, best_san) in enumerate(puzzle):

                # Build single prompt for KINGPT
                prompt =  build_prompt_kingpt(fen)

                # query KINGPT on puzzle position
                raw_response = generate_kingpt_response(model, encode, decode, prompt, device)

                # attempt to parse a valid move from KINGPT response
                board = chess.Board(fen)
                predicted_uci = parse_move(raw_response, board)

                # If no valid move is parsed, FAIL!
                if predicted_uci is None:
                    theme_parse_failed += 1
                    total_parse_failed += 1

                # check if the move is actually accurate
                correct = check_position_accuracy(predicted_uci or '', fen, best_uci, best_san, judge_engine)

                # print position-wide results
                status = 'CORRECT' if correct else 'WRONG'
                parse_note = ' (parse failed)' if predicted_uci is None else ''
                print(f'    Position {pos_idx + 1}:')
                print(f'      FEN:           {fen}')
                print(f'      Best move UCI: {best_uci}')
                print(f'      Raw response:  {raw_response!r}')
                print(f'      Parsed UCI:    {predicted_uci!r}{parse_note}')
                print(f'      Result:        [{status}]')

                # update results if LLM response was correct
                if correct:
                    puzzle_positions_correct += 1
                    theme_positions_correct += 1
                    total_positions_correct += 1
                theme_positions_total += 1
                total_positions += 1
            
            # check if KINGPT solved all puzzle positions
            puzzle_solved = puzzle_positions_correct == len(puzzle)

            # if so, update the results
            if puzzle_solved:
                theme_puzzles_solved += 1
                total_puzzles_solved += 1
            total_puzzles += 1

            # print puzzle-wide results
            puzzle_status = 'SOLVED' if puzzle_solved else 'FAILED'
            print(f'    Puzzle result: [{puzzle_status}] ({puzzle_positions_correct}/{len(puzzle)} positions correct)')

        # Calculate and print theme-wide puzzle + position accuracy
        theme_pos_acc = theme_positions_correct / theme_positions_total * 100 if theme_positions_total > 0 else 0
        theme_puzzle_acc = theme_puzzles_solved / len(puzzles) * 100 if puzzles else 0
        print(f'\n  {theme} summary:')
        print(f'    Puzzles solved:    {theme_puzzles_solved}/{len(puzzles)} ({theme_puzzle_acc:.1f}%)')
        print(f'    Positions correct: {theme_positions_correct}/{theme_positions_total} ({theme_pos_acc:.1f}%)')
        print(f'    Parse failures:    {theme_parse_failed}/{theme_positions_total}')

    # Calculate and print overall puzzle + position accuracy
    overall_pos_acc = total_positions_correct / total_positions * 100 if total_positions > 0 else 0
    overall_puzzle_acc = total_puzzles_solved / total_puzzles * 100 if total_puzzles > 0 else 0
    print(f'\n  OVERALL ({label}):')
    print(f'    Puzzles solved:    {total_puzzles_solved}/{total_puzzles} ({overall_puzzle_acc:.1f}%)')
    print(f'    Positions correct: {total_positions_correct}/{total_positions} ({overall_pos_acc:.1f}%)')
    print(f'    Parse failures:    {total_parse_failed}/{total_positions}')

    # return overall accuracy for display
    return (total_puzzles_solved, total_puzzles, total_positions_correct, total_positions, total_parse_failed)

'''
Helper function which loads, evaluates, and unloads a KINGPT checkpoint.
Same code as evaluate_llm_model but uses a shared meta vocab file and doesn't measure cheating variants.
For details on why no "cheating" type of prompts, please read the very informative and concise comments above evaluate_kingpt_pass()
'''
def evaluate_kingpt_model(model_name: str, ckpt_path: str, puzzles_by_theme: dict, judge_engine: chess.engine.SimpleEngine):

    print(f'\n{"#" * 70}')
    print(f'Loading KINGPT: {model_name} ({ckpt_path})')
    print(f'{"#" * 70}')

    # Load KINGPT instance
    model, encode, decode, device = load_kingpt(ckpt_path, META_PATH)
    print('Model loaded.\n')

    result = evaluate_kingpt_pass(model_name, model, encode, decode, device, puzzles_by_theme, judge_engine)

    # free GPU memory before loading next checkpoint
    del model
    torch.cuda.empty_cache()

    return result


# ============================
# MAIN (MAIN [MAIN {MAIN}])
# ============================

'''
Custom class to write output to both stdout and log file simultaneously.
Pogger == Program logger
'''
class Pogger:
    '''Writes all output to both stdout and a log file simultaneously.'''
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

'''
i have run out of jokes to put in the comments of this program
because my brain is being atrophied due to excessive AI use
help help help help me helpo me hek,pemm me ejsjkoadosln
'''
def main():
    # Set up logging
    LOG_FILE = 'results.txt'
    poggee = Pogger(LOG_FILE)
    sys.stdout = poggee

    # Get a sample of N puzzles from each theme
    print('Sampling puzzles...')
    puzzles_by_theme = fetch_puzzle_sample()
    for theme, (puzzles, _) in puzzles_by_theme.items():
        print(f'  {theme}: {len(puzzles)} puzzle(s) ready')

    # domain expansion: deadly sentencing
    print('\nOpening judge engine (SF18 depth=25)...')
    judge_engine = get_engine()

    # results summary
    summary = {}

    # evaluate KINGPT checkpoints on all puzzles
    for model_name, ckpt_path in KINGPT_MODELS.items():
        summary[model_name] = evaluate_kingpt_model(model_name, ckpt_path, puzzles_by_theme, judge_engine)

    # IMPORTANT: call engine.quit() so you don't use up all of the memory
    judge_engine.quit()

    # Print final summary table across KINGPT variants/configs
    print(f'\n\n{"=" * 95}')
    print('FINAL SUMMARY')
    print(f'{"=" * 95}')
    print(f'{"Model":<48} {"Puzzles":>17} {"Positions":>17} {"Invalid Parses":>10}')
    print(f'{"-" * 95}')
    for model_name, (psolved, ptotal, poscorrect, postotal, parse_failed) in summary.items():
        puzzle_acc = psolved / ptotal * 100 if ptotal > 0 else 0
        pos_acc = poscorrect / postotal * 100 if postotal > 0 else 0
        parse_str = f'{parse_failed}/{postotal}' if parse_failed is not None else 'N/A'
        print(f'{model_name:<48} {f"{psolved}/{ptotal} ({puzzle_acc:.1f}%)":>17} {f"{poscorrect}/{postotal} ({pos_acc:.1f}%)":>17} {parse_str:>10}')

# Make sure stdout doesn't break for fun
if __name__ == '__main__':
    try:
        main()
    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
