ffmpeg -i imgs/inversetruth%04d.png -f segment -strftime 1 videos/forwardtruth%Y-%m-%d_%H-%M-%S.webm
ffmpeg -i imgs/inversemodel%04d.png -f segment -strftime 1 videos/forwardmodel%Y-%m-%d_%H-%M-%S.webm
