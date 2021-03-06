#!/bin/bash

for i in {1..1000}
do
 tw-make tw-treasure_hunter --level 2  --seed $i --output ../resources/train_games_lvl2/game_th_lvl2_$i.ulx
 tw-make tw-treasure_hunter --level 3  --seed $i --output ../resources/train_games_lvl3/game_th_lvl3_$i.ulx
 tw-make tw-treasure_hunter --level 4  --seed $i --output ../resources/train_games_lvl4/game_th_lvl4_$i.ulx
done


for i in {1000..1200}
do
 tw-make tw-treasure_hunter --level 2  --seed $i --output ../resources/test_games_lvl2/game_th_lvl2_$i.ulx
 tw-make tw-treasure_hunter --level 3  --seed $i --output ../resources/test_games_lvl3/game_th_lvl3_$i.ulx
 tw-make tw-treasure_hunter --level 4  --seed $i --output ../resources/test_games_lvl4/game_th_lvl4_$i.ulx
done