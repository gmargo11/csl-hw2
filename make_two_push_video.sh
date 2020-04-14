ffmpeg -i imgs/forward_twopushtruth%05d.png -f segment -strftime 1 videos/forward_twopushtruth%Y-%m-%d_%H-%M-%S.webm
ffmpeg -i imgs/forward_twopushmodel%05d.png -f segment -strftime 1 videos/forward_twopushmodel%Y-%m-%d_%H-%M-%S.webm
ffmpeg -i imgs/inverse_twopushtruth%05d.png -f segment -strftime 1 videos/inverse_twopushtruth%Y-%m-%d_%H-%M-%S.webm
ffmpeg -i imgs/inverse_twopushmodel%05d.png -f segment -strftime 1 videos/inverse_twopushmodel%Y-%m-%d_%H-%M-%S.webm
