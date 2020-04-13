ffmpeg -i imgs/forwardtruth%04d.png -f segment -strftime 1 videos/forwardtruth%Y-%m-%d_%H-%M-%S.webm
ffmpeg -i imgs/forwardmodel%04d.png -f segment -strftime 1 videos/forwardmodel%Y-%m-%d_%H-%M-%S.webm
