ffmpeg -i imgs/inversetruth%04d.png -f segment -strftime 1 videos/inversetruth%Y-%m-%d_%H-%M-%S.webm
ffmpeg -i imgs/inversemodel%04d.png -f segment -strftime 1 videos/inversemodel%Y-%m-%d_%H-%M-%S.webm
