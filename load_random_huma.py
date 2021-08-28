import atari_py

def get_dic():
    
    dic = {}
    with open('random_human.txt') as f:
        for line in f:
            line = line.strip().split(' ')
            game = "_".join(line[:-2])
            assert game in atari_py.list_games()
            random_score = float(line[-2].replace(',', ''))
            human_score = float(line[-1].replace(',', ''))

            assert human_score > random_score
            
            dic[game] = [random_score, human_score]

    assert len(dic) == 57
    return dic

if __name__ == "__main__":

    game_dic = get_dic()
    print(game_dic)