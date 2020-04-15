ffmpeg -pattern_type glob -i 'imgs/forward_twopushtruth*.png' -f segment -strftime 1 videos/forward_twopushtruth%Y-%m-%d_%H-%M-%S.webm
ffmpeg -pattern_type glob -i 'imgs/forward_twopushmodel*.png' -f segment -strftime 1 videos/forward_twopushmodel%Y-%m-%d_%H-%M-%S.webm
ffmpeg -pattern_type glob -i 'imgs/inverse_twopushtruth*.png' -f segment -strftime 1 videos/inverse_twopushtruth%Y-%m-%d_%H-%M-%S.webm
ffmpeg -pattern_type glob -i 'imgs/inverse_twopushmodel*.png' -f segment -strftime 1 videos/forward_twopushmodel%Y-%m-%d_%H-%M-%S.webm
