import os

res_path = os.path.dirname(os.path.realpath(__file__))

debug_game = "/game_th_lvl2_simple.ulx"
def_verbs = "/words_verbs_twcore.txt"
def_objs = "/words_objs_auto.txt"
def_badact = "/bad_actions.txt"

DEFAULT_PATHS = {
    "env_name": os.path.join(res_path + debug_game),
    "path_verbs": os.path.join(res_path + def_verbs),
    "path_objs": os.path.join(res_path + def_objs),
    "path_badact": os.path.join(res_path + def_badact),
}

train_dir = [
    os.path.join(res_path + "/train_games_lvl1"),
    os.path.join(res_path + "/train_games_lvl2"),
    os.path.join(res_path + "/train_games_lvl3"),
    os.path.join(res_path + "/train_games_lvl4"),
]
test_dir = [
    os.path.join(res_path + "/test_games_lvl1"),
    os.path.join(res_path + "/test_games_lvl2"),
    os.path.join(res_path + "/test_games_lvl3"),
    os.path.join(res_path + "/test_games_lvl4"),
]
