# style-transfer-ibb
Style transfer solution project.
This project is able to transfer styles from one image to another. The core logic is integrated in `style_transfer.py`. Additionally, a realtime solution has been implemented at `realtime.py`.

![ibb_proj_showcase](https://user-images.githubusercontent.com/23532665/149754163-ec2f2c2d-c37d-4896-913d-22b80ba8656d.gif)

Alpha and gamma parameters can be controlled by `-a|g {val} or --a|g={value}`.

Basic usage for style_transfer: `python3 style_transfer.py --src=poses/no_makeup.jpeg --ref=poses/makeup_tryon.jpeg -a 1 -g 0.4 --path=logs/test`. 

Basic usage for realtime: `python3 realtime.py -a 1 -g 0.5`. You can tweak the parameters via the GUI but it might be buggy depending on ur GPU so you can start with pre-set value. Additionally, you can specify the reference image (the image to copy the style from) via `--ref={img dir}`. 

Errors are not caught. Usually if there's an error, it has to do with the face detection.


_This page will be updated with more information in the upcoming days._
