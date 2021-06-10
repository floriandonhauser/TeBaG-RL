#!/bin/bash
tw-make tw-simple --rewards dense --goal detailed --seed 2021 --output ../resources/game_simple_densedetailed.ulx
tw-make tw-simple --rewards dense --goal brief --seed 2021 --output ../resources/game_simple_densebrief.ulx

tw-make tw-coin_collector --level 3  --seed 2021 --output ../resources/game_coin_3.ulx
tw-make tw-treasure_hunter  --level 3 --seed 2021 --output ../resources/game_treasure_3.ulx
tw-make tw-coin_collector --level 5  --seed 2021 --output ../resources/game_coin_5.ulx
tw-make tw-treasure_hunter  --level 5 --seed 2021 --output ../resources/game_treasure_5.ulx