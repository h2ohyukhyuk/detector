### Code
https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch

### Code with train code
https://github.com/ayooshkathuria/pytorch-yolo-v3

### Blog Post
https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/

### yolov3 설명 잘된 블로그
https://herbwood.tistory.com/21
  - bbox 좌표를 만드는 역과정을 GT에 적용하여 GT를 만들고 L1 loss를 사용
    - bbox sig(tx),sig(ty),exp(tw),exp(th) L2 loss -> box tx,ty,tw,th L1 loss
  - GT와 가장 가까운 anchor box와 관련된 bbox관 loss 계산
  - objectness score 는 모든 grid pixel에 loss 계산
  - multi-class loss를 softmax -> bce(sigmoid), bbox에 여러클래스가 있을수 있기때문에.
  - prediction across scales
    - 416 input -> 52, 26, 13 feature maps, upsampling, concat -> large, medium, small output
  - Darknet-53 backbone, 53 layers with shortcut connections
