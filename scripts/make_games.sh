#!/bin/bash
tw-make tw-simple --rewards dense --goal detailed --seed 2021 --output ../resources/game_simple_densedetailed.ulx
tw-make tw-simple --rewards dense --goal brief --seed 2021 --output ../resources/game_simple_densebrief.ulx

for i in {1..10}
do
   tw-make tw-treasure_hunter --level 1  --seed $i --output ../resources/gen_games/game_coin_lvl1_$i.ulx
   tw-make tw-treasure_hunter --level 2  --seed $i --output ../resources/gen_games/game_coin_lvl2_$i.ulx
   tw-make tw-treasure_hunter --level 3  --seed $i --output ../resources/gen_games/game_coin_lvl3_$i.ulx
done